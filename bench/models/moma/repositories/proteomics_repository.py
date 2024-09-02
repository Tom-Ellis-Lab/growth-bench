import abc

from bench.models.moma.gateways import proteomics_gateway
from bench.models.moma.entities import omics, proteomics


class ProteomicsRepositoryInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> list[proteomics.ProteinAbundanceProfile]:
        pass


class ProteomicsRepository(ProteomicsRepositoryInterface):
    def __init__(self, gateway: proteomics_gateway.ProteomicsGateway):
        self.gateway = gateway

    def get(self) -> list[proteomics.ProteinAbundanceProfile]:

        data = self.gateway.get()
        result = []
        id_counter = 0
        for protein_data in data:
            protein = proteomics.Protein(id=protein_data.protein_id)
            condition = omics.GeneKnockout(
                standard_name=protein_data.ko_gene_standard_name,
                systematic_name=protein_data.ko_gene_systematic_name,
                expression_level=protein_data.ko_gene_expression_level,
            )
            metada = proteomics.ProteomicsMetadata(
                injection_nr=protein_data.injection_nr,
                well_nr=protein_data.well_nr,
                batch_nr=protein_data.batch_nr,
                is_quality_control=protein_data.is_quality_control,
                is_his3delta_control=protein_data.is_his3delta_control,
            )
            id_counter = id_counter + 1
            result.append(
                proteomics.ProteinAbundanceProfile(
                    id=str(id_counter),
                    protein=protein,
                    condition=condition,
                    abundance_level=protein_data.protein_abundance_level,
                    metadata=metada,
                )
            )
        return result
