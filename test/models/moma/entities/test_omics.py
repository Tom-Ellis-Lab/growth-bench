import pytest

from bench.models.moma.entities import omics


class TestMolecularEntity:
    @pytest.fixture
    def test_init(self):
        observed = omics.MolecularEntity(id="entity_id")
        assert observed.id == "entity_id"


class TestGeneKnockout:
    @pytest.fixture
    def test_init(self):
        observed = omics.GeneKnockout(
            standard_name="gene_standard_name",
            systematic_name="gene_systematic_name",
            expression_level=1.0,
        )
        assert observed.name == "gene_standard_name"
        assert observed._systematic_name == "gene_systematic_name"
        assert observed._expression_level == 1.0

    def test_init_no_standard_name(self):
        observed = omics.GeneKnockout(
            systematic_name="gene_systematic_name", expression_level=1.0
        )
        assert observed.name == "gene_systematic_name"
        assert observed._standard_name is None
        assert observed._systematic_name == "gene_systematic_name"

    def test_init_no_systematic_name(self):
        observed = omics.GeneKnockout(
            standard_name="gene_standard_name", expression_level=1.0
        )
        assert observed._standard_name == "gene_standard_name"
        assert observed.name == "gene_standard_name"
        assert observed._systematic_name is None

    def test_init_no_expression_level(self):
        observed = omics.GeneKnockout(
            standard_name="gene_standard_name", systematic_name="gene_systematic_name"
        )
        assert observed._expression_level is None
