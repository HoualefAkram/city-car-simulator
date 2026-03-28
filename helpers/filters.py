from data_models.base_tower import BaseTower
from data_models.ng_ran_report import NGRANReport
from utils.wave_utils import WaveUtils


class Filters:

    @staticmethod
    def top_k_towers(
        all_bs: list[BaseTower],
        report: NGRANReport,
        k: int = 4,
        rsrp_weight: float = 0.5,
        rsrq_weight: float = 0.5,
    ):

        scores: dict[int, float] = {}

        for bs in all_bs:
            bs_id = bs.id

            norm_rsrp = WaveUtils.normalize_rsrp_index(
                report.rsrp_values.get(bs_id, 0), bs.radio
            )
            norm_rsrq = WaveUtils.normalize_rsrq_index(
                report.rsrq_values.get(bs_id, 0), bs.radio
            )

            scores[bs_id] = (norm_rsrp * rsrp_weight) + (norm_rsrq * rsrq_weight)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        top_k_ids = [item[0] for item in sorted_scores[:k]]

        tower_lookup = {bs.id: bs for bs in all_bs}
        top_k_towers = [tower_lookup[t_id] for t_id in top_k_ids]

        return top_k_towers
