import pytest
import pandas as pd
from bench.models.moma.preprocessing_utils import mappers
from unittest.mock import MagicMock


@pytest.fixture
def gene_names_gateway():
    # Mock the gene names gateway to return sample data
    gateway = MagicMock()
    gateway.get.return_value = pd.DataFrame(
        {
            "systematic_name": ["SystematicA", "SystematicB"],
            "standard_name": ["GeneA", "GeneB"],
        }
    )
    gateway.get_standard_name.return_value = "standard_name"
    gateway.get_systematic_name.return_value = "systematic_name"
    return gateway


@pytest.fixture
def gene_names_mapper():
    return mappers.GeneNamesMapper()


class TestGeneNamesMapper:

    def test_create_mapping(self, gene_names_gateway, gene_names_mapper):
        mapping_dict = gene_names_mapper.create_mapping(gene_names_gateway)

        # Check if the mapping dictionary has the expected keys
        assert mapping_dict["GeneA"] == "SystematicA"
        assert mapping_dict["GeneB"] == "SystematicB"
        assert mapping_dict["SystematicA"] == "SystematicA"
        assert mapping_dict["SystematicB"] == "SystematicB"

        # Ensure the mapping dict is set after creation
        assert gene_names_mapper._mapping_dict is not None

    def test_map(self, gene_names_gateway, gene_names_mapper):
        # Create mapping first
        gene_names_mapper.create_mapping(gene_names_gateway)

        # Prepare sample data
        df = pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB", "GeneC"],
                "value": [1, 2, 3],
            }
        )

        # Map gene names
        mapped_df = gene_names_mapper.map(
            df, target_col="gene", new_col="mapped_gene", fillna_col="gene"
        )

        # Expected result should map GeneA to SystematicA and GeneB to SystematicB
        expected_df = pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB", "GeneC"],
                "value": [1, 2, 3],
                "mapped_gene": [
                    "SystematicA",
                    "SystematicB",
                    "GeneC",
                ],  # GeneC should remain unchanged
            }
        )

        pd.testing.assert_frame_equal(mapped_df, expected_df)

    def test_map_without_mapping_dict(self, gene_names_mapper):
        df = pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB"],
                "value": [1, 2],
            }
        )

        # Should raise an error if the mapping dict is not set
        with pytest.raises(ValueError, match="Mapping dict not set"):
            gene_names_mapper.map(
                df, target_col="gene", new_col="mapped_gene", fillna_col="gene"
            )


class TestMapperFactory:

    @pytest.fixture
    def mapper_factory(self):
        return mappers.MapperFactory()

    def test_create_mapper(self, mapper_factory):
        mapper = mapper_factory.create_mapper(mappers.MapperType.GENE_NAMES)
        assert isinstance(mapper, mappers.GeneNamesMapper)

    def test_create_mapper_invalid_type(self, mapper_factory):
        with pytest.raises(ValueError, match="Unknown mapper type: invalid_type"):
            mapper_factory.create_mapper("invalid_type")
