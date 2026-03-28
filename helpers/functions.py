from math import exp


class Functions:
    @staticmethod
    def softmax(all_values: list[float], index: int) -> float:
        """Returns only the softmax output of the index"""
        max_val = max(all_values)
        numerator = exp(all_values[index] - max_val)
        denominator = sum(exp(v - max_val) for v in all_values)
        if denominator == 0:
            return 0
        return numerator / denominator

    @staticmethod
    def softmax_all(all_values: list[float]) -> list[float]:
        """Returns the softmax output of all_values"""
        return [
            Functions.softmax(all_values=all_values, index=i)
            for i in range(len(all_values))
        ]
