"""Shared event-resolved hybrid simulation; models supply physical event surfaces."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class GuardEvent:
    name: str
    function: object  # function(state) = 0 at a physical switching surface
    modes: tuple
    direction: float = 0.
    domain_face: tuple = ()  # (axis, -1 lower / +1 upper) handled by this reset event


def _normalize_guard_face(state, event, lower, upper):
    """Canonicalize only a located boundary root, within eight float64 epsilons."""
    if not event.domain_face:
        return state, None
    axis, sign = event.domain_face
    boundary = lower[axis] if sign == -1 else upper[axis]
    tolerance = 8*np.finfo(float).eps*max(1., abs(boundary))
    error = state[axis]-boundary
    if abs(error) > tolerance:
        raise RuntimeError('Located guard root is inconsistent with its declared domain face')
    result = state.copy()
    result[axis] = boundary
    return result, dict(axis=axis, raw=float(state[axis]), canonical=float(boundary), error=float(error))


def simulate(problem, controller, initial_state, initial_mode, *, duration, feedback_dt,
             events=(), failure=None, goal=None, disturbance=None, coordinates=None,
             flow=None, max_step=.01, rtol=1e-9, atol=1e-11, fixed_axes=None,
             segment_audit=None):
    """Controls are queried after resets; integrate up to first event or sample.

    failure/goal are scalar continuous negative-inside functions of (state,mode).
    Guard events must cover all active physical switching surfaces. Their
    functions and directions are model inputs, not inferred from Boolean guards.
    Starts exactly on a guard require a valid departing mode; unresolved
    zero-duration events stop as zeno_contact instead of advancing through them.
    Numerical ODE event detection is not a continuous safety certificate.
    """
    if not np.isfinite(duration+feedback_dt) or duration <= 0 or feedback_dt <= 0:
        raise ValueError("Expected positive finite simulation durations")
    model = problem.configured_model()
    from .coordinates import IdentityCoordinates
    chart = coordinates or IdentityCoordinates()
    axes = tuple(np.asarray(v) for v in problem.grid.coordinate_vectors)
    lower = chart.decode(np.array([v[0] for v in axes]))
    upper = chart.decode(np.array([v[-1] for v in axes]))
    x = np.asarray(initial_state, dtype=float).copy()
    if x.shape != lower.shape or not np.isfinite(x).all():
        raise ValueError("Invalid initial state")
    if not isinstance(initial_mode, (int, np.integer)) or not 0 <= initial_mode < len(model.modes):
        raise ValueError("Invalid initial mode")
    q, t, records, resets, audits, endpoints, root_corrections = initial_mode, 0., [], [], [], [], []
    status = "deadline"
    def current_status(state, mode):
        if failure is not None and failure(state, mode) <= 0:
            return "failure"
        if np.any(state < lower) or np.any(state > upper):
            return "domain_exit"
        return None
    while t < duration-1e-12:
        status_now = current_status(x, q)
        if status_now:
            status = status_now
            break
        decision = controller(x.copy(), q, duration-t)
        reset_failure = None
        replay_x, replay_q = x.copy(), q
        for stage_index, edge_index in decision.transitions:
            edge = model.stages[stage_index].transitions[edge_index]
            if edge.source != replay_q or not bool(np.asarray(edge.guard(chart.encode(replay_x)))):
                raise ValueError("Controller requested an ineligible transition")
            replay_x = chart.decode(np.asarray(edge.reset(chart.encode(replay_x)), dtype=float))
            if edge.boundary == "clip":
                replay_x = np.clip(replay_x, lower, upper)
            replay_q = edge.destination
            reset_failure = current_status(replay_x, replay_q)
            if reset_failure:
                break
        if reset_failure:
            x, q, status = replay_x, replay_q, reset_failure
            break
        if replay_q != decision.mode or not np.allclose(chart.encode(replay_x),
                chart.encode(decision.state), rtol=0, atol=1e-12):
            raise ValueError("Controller state/mode does not match its reset trace")
        if decision.transitions:
            resets.append(dict(time=t, source=q, destination=decision.mode,
                               transitions=decision.transitions, before=x.tolist(),
                               after=np.asarray(decision.state).tolist()))
        x, q = np.asarray(decision.state), decision.mode
        status_now = current_status(x, q)
        if status_now:
            status = status_now
            break
        if goal is not None and goal(x, q) <= 0:
            status = "goal_reached"
            break
        dynamics = model.modes[q].dynamics
        u = np.asarray(decision.control)
        if not np.isfinite(u).all() or u.shape != np.asarray(dynamics.control_space.lo).shape or np.any(
                u < np.asarray(dynamics.control_space.lo)) or np.any(u > np.asarray(dynamics.control_space.hi)):
            raise ValueError("Controller returned an inadmissible input")
        if disturbance is None and np.asarray(dynamics.disturbance_space.lo).size:
            raise ValueError("Supply a disturbance policy for disturbed simulation")
        def rhs(time, state):
            d = np.zeros(0) if disturbance is None else np.asarray(disturbance(time, state, q))
            if (d.shape != np.asarray(dynamics.disturbance_space.lo).shape or not np.isfinite(d).all()
                    or np.any(d < np.asarray(dynamics.disturbance_space.lo))
                    or np.any(d > np.asarray(dynamics.disturbance_space.hi))):
                raise ValueError("Inadmissible disturbance")
            if flow is not None:
                return np.asarray(flow(state, q, u, d, time), dtype=float)
            computational = np.asarray(dynamics(chart.encode(state), u, d, time), dtype=float)
            return computational*chart.inverse_jacobian(state)
        functions, names = [], []
        def add(name, fn, direction):
            fn.terminal, fn.direction = True, direction
            functions.append(fn)
            names.append(name)
        # Failure gets priority for simultaneous goal/failure contact.
        if failure is not None:
            add("failure", lambda time, state: failure(state, q), -1)
        for axis in range(len(x)):
            if fixed_axes is not None and axis in fixed_axes(q):
                continue
            covered = {e.domain_face for e in events if q in e.modes and e.domain_face}
            if (axis, -1) not in covered:
                add("domain_exit", lambda time, state, a=axis: state[a]-lower[a], -1)
            if (axis, 1) not in covered:
                add("domain_exit", lambda time, state, a=axis: upper[a]-state[a], -1)
        if goal is not None:
            add("goal_reached", lambda time, state: goal(state, q), -1)
        for event in events:
            if q in event.modes:
                add("guard:"+event.name, lambda time, state, e=event: e.function(state), event.direction)
        end = min(duration, t+feedback_dt)
        solution = solve_ivp(rhs, (t, end), x, events=functions or None,
                             method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                             max_step=min(feedback_dt, max_step))
        if not solution.success:
            raise RuntimeError(solution.message)
        new_t, new_x = float(solution.t[-1]), solution.y[:, -1]
        if new_t-t < 1e-12:
            status = "zeno_contact"
            break
        records.append((t, x.copy(), q, u.copy(), decision.value,
                        decision.continuous_intervention, decision.discrete_intervention, decision.safe_candidate))
        if segment_audit is not None:
            audit = dict(segment_audit(solution, t, new_t, q, u))
            audits.append(audit)
        else:
            audit = {}
        hit = next((names[i] for i, times in enumerate(solution.t_events) if len(times)), None)
        if hit and hit.startswith('guard:'):
            event = next(e for e in events if q in e.modes and 'guard:'+e.name == hit)
            new_x, correction = _normalize_guard_face(new_x, event, lower, upper)
            if correction is not None and correction['error'] != 0:
                root_corrections.append(dict(time=new_t, event=event.name, **correction))
        endpoints.append(new_x.copy())
        t, x = new_t, new_x
        if hit and not hit.startswith("guard:"):
            status = current_status(x, q) or hit
            break
        if audit.get('clearance_lower_bound', 1.) <= 0:
            status = 'clearance_unresolved'
            break
    return dict(status=status, final_time=t, final_state=x, final_mode=q,
        time=np.array([r[0] for r in records]+[t]),
        state=np.array([r[1] for r in records]+[x]),
        mode=np.array([r[2] for r in records]+[q]),
        control=np.array([r[3] for r in records]),
        value=np.array([r[4] for r in records]),
        continuous_intervention=np.array([r[5] for r in records]),
        discrete_intervention=np.array([r[6] for r in records]), resets=resets,
        safe_candidate=np.array([r[7] for r in records], dtype=bool),
        root_corrections=root_corrections,
        segment_audits=audits, segment_end_state=np.array(endpoints),
        time_to_go=np.array([duration-r[0] for r in records]))
