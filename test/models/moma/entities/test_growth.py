import pytest

from bench.models.moma.entities import growth, omics


class TestGrowthMedium:

    @pytest.fixture
    def expected_values(self):
        result = ["SC", "SM", "YPD", "SD"]
        return result

    @pytest.fixture
    def expected_names(self):
        result = ["SC", "SM", "YPD", "SD"]
        return result

    def test_values(self, expected_values, expected_names):
        for growth_medium in growth.GrowthMedium:
            assert growth_medium.value in expected_values
            assert growth_medium.name in expected_names


class TestGrowthRateData:

    @pytest.fixture
    def input(self):
        result = {
            "growth_rate": 1.2,
            "medium": growth.GrowthMedium.SD,
            "ko_gene_standard_name": "gene_standard_name",
            "ko_gene_systematic_name": "gene_systematic_name",
        }
        return result

    def test_init(self, input):
        observed = growth.GrowthRateData(**input)
        assert observed.growth_rate == input["growth_rate"]
        assert observed.medium == input["medium"]
        assert observed.ko_gene_standard_name == input["ko_gene_standard_name"]
        assert observed.ko_gene_systematic_name == input["ko_gene_systematic_name"]

    def test_init_no_ko_gene_standard_name(self, input):
        input["ko_gene_standard_name"] = None
        observed = growth.GrowthRateData(**input)
        assert observed.ko_gene_standard_name is None

    def test_init_no_ko_gene_systematic_name(self, input):
        input["ko_gene_systematic_name"] = None
        observed = growth.GrowthRateData(**input)
        assert observed.ko_gene_systematic_name is None

    def test_init_no_ko_gene_standard_name_and_no_ko_gene_systematic_name(self, input):
        input["ko_gene_standard_name"] = None
        input["ko_gene_systematic_name"] = None
        with pytest.raises(ValueError):
            growth.GrowthRateData(**input)


class TestGrowthRateMeasurement:

    @pytest.fixture
    def input(self):
        result = {
            "id": "1",
            "growth_rate": 1.2,
            "condition": omics.GeneKnockout(
                standard_name="gene_standard_name",
                systematic_name="gene_systematic_name",
            ),
            "medium": growth.GrowthMedium.SD,
        }
        return result

    def test_init(self, input):
        observed = growth.GrowthRateMeasurement(**input)
        assert observed.growth_rate == input["growth_rate"]
        assert observed.condition == input["condition"]


class TestGrowthRateDataset:

    @pytest.fixture
    def input(self):
        result = [
            growth.GrowthRateMeasurement(
                id="1",
                growth_rate=1.2,
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_A",
                    systematic_name="gene_systematic_A",
                ),
                medium=growth.GrowthMedium.SD,
            ),
            growth.GrowthRateMeasurement(
                id="2",
                growth_rate=3.4,
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_B",
                    systematic_name="gene_systematic_B",
                ),
                medium=growth.GrowthMedium.SD,
            ),
        ]
        return result

    @pytest.fixture
    def dataset(self, input):
        return growth.GrowthRateDataset(data=input)

    def test_get_by_id(self, dataset, input):
        observed = dataset.get_by_id("1")
        assert observed == input[0]
        with pytest.raises(ValueError):
            dataset.get_by_id("3")

    def test_group_by_id(self, dataset, input):
        observed = dataset.group_by_id()
        assert len(observed) == 2
        assert len(observed["1"]) == 1
        assert len(observed["2"]) == 1
        assert observed["1"][0] == input[0]
        assert observed["2"][0] == input[1]

    def test_group_by_condition(self, dataset, input):
        observed = dataset.group_by_condition()
        assert len(observed) == 2
        assert len(observed["gene_standard_A"]) == 1
        assert len(observed["gene_standard_B"]) == 1
        assert observed["gene_standard_A"][0] == input[0]
        assert observed["gene_standard_B"][0] == input[1]

    def test_group_by_medium(self, dataset, input):
        observed = dataset.group_by_medium()
        assert len(observed) == 1
        assert len(observed["SD"]) == 2
        assert observed["SD"][0] == input[0]
        assert observed["SD"][1] == input[1]
