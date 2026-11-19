import math
from collections.abc import Mapping
from numbers import Real

DEFAULT_ALPHA = 0.05
HARM_DIRECTIONS = ('negative', 'positive')


def _require_number(value, name):
    # bool is a subclass of int, but it is not a valid number here
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    return value


def _validate_result(result, name, require_harm_direction):
    if not isinstance(result, Mapping):
        raise TypeError(f"{name} must be a dict, got {type(result).__name__}.")
    if len(result) == 0:
        raise ValueError(f"{name} is empty.")

    for key in ('p_value', 'effect'):
        if key not in result:
            raise ValueError(f"{name} is missing required key '{key}'.")

    clean = dict(result)  # copy, so the caller's dict is not changed

    clean['p_value'] = _require_number(result['p_value'], f"{name}['p_value']")
    if not 0.0 <= clean['p_value'] <= 1.0:
        raise ValueError(f"{name}['p_value'] must be in [0, 1], got {clean['p_value']}.")

    clean['effect'] = _require_number(result['effect'], f"{name}['effect']")

    clean['alpha'] = _require_number(result.get('alpha', DEFAULT_ALPHA), f"{name}['alpha']")
    if not 0.0 < clean['alpha'] < 1.0:
        raise ValueError(f"{name}['alpha'] must be in (0, 1), got {clean['alpha']}.")

    if require_harm_direction:
        if 'harm_direction' not in result:
            raise ValueError(f"{name} is missing required key 'harm_direction'.")
        if result['harm_direction'] not in HARM_DIRECTIONS:
            raise ValueError(
                f"{name}['harm_direction'] must be one of {HARM_DIRECTIONS}, "
                f"got {result['harm_direction']!r}."
            )

    return clean


def validate_ship_decision_inputs(primary_result, guardrail_results):
    primary = _validate_result(primary_result, 'primary_result', require_harm_direction=False)

    if guardrail_results is None:
        guardrail_results = []
    if not isinstance(guardrail_results, (list, tuple)):
        raise TypeError(
            f"guardrail_results must be a list, got {type(guardrail_results).__name__}."
        )

    guardrails = [
        _validate_result(g, f"guardrail_results[{i}]", require_harm_direction=True)
        for i, g in enumerate(guardrail_results)
    ]
    return primary, guardrails


def ship_decision(primary_result, guardrail_results):
    primary_result, guardrail_results = validate_ship_decision_inputs(
        primary_result, guardrail_results
    )

    result = {'ship': True,
            'reason': 'ship'}

    if primary_result['p_value'] >= primary_result['alpha']:
        result['ship'] = False
        result['reason'] = "primary metric not significant"
        return result

    if primary_result['effect'] <= 0:
        result['ship'] = False
        result['reason'] = "primary metric effect not positive"
        return result

    for i, guardrail in enumerate(guardrail_results):
        if guardrail['p_value'] < guardrail['alpha']:

            harmed = (
                (guardrail['harm_direction'] == 'negative' and guardrail['effect'] < 0)
                or (guardrail['harm_direction'] == 'positive' and guardrail['effect'] > 0)
            )
            if harmed:
                result['ship'] = False
                result['reason'] = f"guardrail {i} harmed"
                return result

    return result
