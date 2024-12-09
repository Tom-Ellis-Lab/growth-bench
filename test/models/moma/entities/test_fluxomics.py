import pytest

from bench.models.moma.entities import fluxomics, omics


class TestMetabolicReaction:

    def test_init(self):
        observed = fluxomics.MetabolicReaction(id="reaction_id")
        assert observed.id == "reaction_id"


class TestMetabolicReactionData:

    def test_init(self):
        observed = fluxomics.MetabolicReactionData(
            reaction_id="reaction_id",
            flux_rate=1.0,
            ko_gene_standard_name="gene_standard_name",
        )
        assert observed.reaction_id == "reaction_id"
        assert observed.flux_rate == 1.0
        assert observed.ko_gene_standard_name == "gene_standard_name"


class TestMetabolicFLuxProfile:

    def test_init(self):
        observed = fluxomics.MetabolicFLuxProfile(
            id="1",
            reaction=fluxomics.MetabolicReaction(id="reaction_id"),
            condition=omics.GeneKnockout(
                standard_name="gene_standard_name",
                systematic_name="gene_systematic_name",
                expression_level=1.0,
            ),
            flux_rate=1.0,
        )
        assert observed.id == "1"
        assert observed.molecular_entity.id == "reaction_id"
        assert observed.value == 1.0
        assert observed.condition.name == "gene_standard_name"
        with pytest.raises(NotImplementedError):
            observed.quality_control
        with pytest.raises(NotImplementedError):
            observed.his3delta_control


class TestFluxomics:

    @pytest.fixture
    def profiles(self):
        result = [
            fluxomics.MetabolicFLuxProfile(
                id="1",
                reaction=fluxomics.MetabolicReaction(id="reaction_A"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_A",
                    systematic_name="gene_systematic_name_A",
                    expression_level=0.0,
                ),
                flux_rate=1.0,
            ),
            fluxomics.MetabolicFLuxProfile(
                id="2",
                reaction=fluxomics.MetabolicReaction(id="reaction_B"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_B",
                    systematic_name="gene_systematic_name_B",
                    expression_level=2.0,
                ),
                flux_rate=2.0,
            ),
        ]
        return result

    @pytest.fixture
    def fluxomics_data(self, profiles):
        return fluxomics.Fluxomics(data=profiles)

    def test_return_as_list(self, fluxomics_data, profiles):
        observed = fluxomics_data.return_as_list()
        assert observed == profiles

    def test_group_by_id(self, fluxomics_data):
        observed = fluxomics_data.group_by_id()
        assert len(observed) == 2
        assert len(observed["reaction_A"]) == 1
        assert len(observed["reaction_B"]) == 1
        assert observed["reaction_A"][0].molecular_entity.id == "reaction_A"
        assert observed["reaction_A"][0].value == 1.0
        assert observed["reaction_A"][0].condition.name == "gene_standard_name_A"
        assert observed["reaction_B"][0].molecular_entity.id == "reaction_B"
        assert observed["reaction_B"][0].value == 2.0
        assert observed["reaction_B"][0].condition.name == "gene_standard_name_B"

    def test_get_by_id(self, fluxomics_data, profiles):
        observed = fluxomics_data.get_by_id("1")
        assert observed == profiles[0]
        with pytest.raises(ValueError):
            fluxomics_data.get_by_id("3")

    def test_group_by_condition(self, fluxomics_data):

        observed = fluxomics_data.group_by_condition()
        assert len(observed) == 2
        assert len(observed["gene_standard_name_A"]) == 1
        assert len(observed["gene_standard_name_B"]) == 1
        assert observed["gene_standard_name_A"][0].molecular_entity.id == "reaction_A"
        assert observed["gene_standard_name_A"][0].value == 1.0
        assert (
            observed["gene_standard_name_A"][0].condition.name == "gene_standard_name_A"
        )
        assert observed["gene_standard_name_B"][0].molecular_entity.id == "reaction_B"
        assert observed["gene_standard_name_B"][0].value == 2.0
        assert (
            observed["gene_standard_name_B"][0].condition.name == "gene_standard_name_B"
        )

    def test_set(self, profiles):
        new_data = [
            fluxomics.MetabolicFLuxProfile(
                id="3",
                reaction=fluxomics.MetabolicReaction(id="reaction_C"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_C",
                    systematic_name="gene_systematic_name_C",
                    expression_level=3.0,
                ),
                flux_rate=3.0,
            ),
            fluxomics.MetabolicFLuxProfile(
                id="4",
                reaction=fluxomics.MetabolicReaction(id="reaction_D"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_D",
                    systematic_name="gene_systematic_name_D",
                    expression_level=4.0,
                ),
                flux_rate=4.0,
            ),
        ]
        fluxomics_data = fluxomics.Fluxomics(data=profiles)
        fluxomics_data.set(new_data)
        observed = fluxomics_data.return_as_list()
        assert observed == new_data
