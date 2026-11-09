def ship_decision(primary_result, guardrail_results):
    if len(primary_result) == 0:
        raise ValueError("primary_result is empty.")

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

            if guardrail['harm_direction'] == 'negative':
                if guardrail['effect'] < 0:
                    result['ship'] = False
                    result['reason'] = f"guardrail {i} harmed"
                    return result
            
            elif guardrail['harm_direction'] == 'positive':
                if guardrail['effect'] > 0:
                    result['ship'] = False
                    result['reason'] = f"guardrail {i} harmed"
                    return result

            else:
                raise ValueError("Invalid harm direction.")
        
    return result