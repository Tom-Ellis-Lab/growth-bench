import abc

from bench.models.moma.gateways import transcriptomics_gateway
from bench.models.moma.entities import omics, transcriptomics


class TranscriptomicsRepositoryInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> list[transcriptomics.TranscriptExpressionProfile]:
        pass


class TranscriptomicsRepository(TranscriptomicsRepositoryInterface):
    def __init__(self, gateway: transcriptomics_gateway.TranscriptomicsGateway):
        self.gateway = gateway

    def get(self) -> list[transcriptomics.TranscriptExpressionProfile]:
        data = self.gateway.get()
        result = []
        id_counter = 0
        for transcript_profile in data:
            id_counter = id_counter + 1
            profile = transcriptomics.TranscriptExpressionProfile(
                id=str(id_counter),
                transcript=transcriptomics.Transcript(
                    id=transcript_profile.transcript_id
                ),
                condition=omics.GeneKnockout(
                    standard_name=transcript_profile.ko_gene_standard_name
                ),
                expression_level=transcript_profile.transcript_expression_level,
            )
            result.append(profile)
        return result
