import pytest

from bench.models.moma.entities import omics, proteomics


class TestProtein:

    def test_init(self):
        observed = proteomics.Protein(id="protein_id")
        assert observed.id == "protein_id"


class TestProteinProfileData:

    def test_init(self):
        observed = proteomics.ProteinProfileData(
            protein_id="protein_id",
            protein_abundance_level=1.0,
            injection_nr=1,
            well_nr=1,
            batch_nr="batch_nr",
            is_quality_control=True,
            is_his3delta_control=True,
            ko_gene_systematic_name="gene_systematic_name",
            ko_gene_standard_name="gene_standard_name",
            ko_gene_expression_level=1.0,
        )
        assert observed.protein_id == "protein_id"
        assert observed.protein_abundance_level == 1.0
        assert observed.injection_nr == 1
        assert observed.well_nr == 1
        assert observed.batch_nr == "batch_nr"
        assert observed.is_quality_control is True
        assert observed.is_his3delta_control is True
        assert observed.ko_gene_systematic_name == "gene_systematic_name"
        assert observed.ko_gene_standard_name == "gene_standard_name"
        assert observed.ko_gene_expression_level == 1.0

    def test_init_no_ko_gene_expression_level(self):
        observed = proteomics.ProteinProfileData(
            protein_id="protein_id",
            protein_abundance_level=1.0,
            injection_nr=1,
            well_nr=1,
            batch_nr="batch_nr",
            is_quality_control=True,
            is_his3delta_control=True,
            ko_gene_systematic_name="gene_systematic_name",
            ko_gene_standard_name="gene_standard_name",
        )
        assert observed.ko_gene_expression_level is None


class TestProteomicsMetadata:

    def test_init(self):
        observed = proteomics.ProteomicsMetadata(
            injection_nr=1,
            well_nr=1,
            batch_nr="batch_nr",
            is_quality_control=True,
            is_his3delta_control=True,
        )
        assert observed.injection_nr == 1
        assert observed.well_nr == 1
        assert observed.batch_nr == "batch_nr"
        assert observed.is_quality_control is True
        assert observed.is_his3delta_control is True


class TestProteinAbundanceProfile:

    @pytest.fixture
    def protein(self):
        result = proteomics.Protein(id="protein_id")
        return result

    @pytest.fixture
    def experimental_condition(self):
        result = omics.GeneKnockout(
            standard_name="gene_standard_name",
            systematic_name="gene_systematic_name",
            expression_level=1.0,
        )
        return result

    @pytest.fixture
    def proteomics_metadata(self):
        result = proteomics.ProteomicsMetadata(
            injection_nr=1,
            well_nr=1,
            batch_nr="batch_nr",
            is_quality_control=True,
            is_his3delta_control=True,
        )
        return result

    def test_init(self, protein, experimental_condition, proteomics_metadata):
        observed = proteomics.ProteinAbundanceProfile(
            id="1",
            protein=protein,
            condition=experimental_condition,
            abundance_level=1.0,
            metadata=proteomics_metadata,
        )
        assert observed.id == "1"
        assert observed.molecular_entity == protein
        assert observed.value == 1.0
        assert observed.condition == experimental_condition
        assert observed.metadata_details == proteomics_metadata
        assert observed.quality_control is True
        assert observed.his3delta_control is True


class TestProteomics:

    @pytest.fixture
    def data(self):
        result = [
            proteomics.ProteinAbundanceProfile(
                id="1",
                protein=proteomics.Protein(id="protein_a"),
                condition=omics.GeneKnockout(
                    standard_name="standard_name_a",
                    systematic_name="systematic_name_a",
                    expression_level=1.0,
                ),
                abundance_level=1.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=1,
                    batch_nr="hrp1",
                    is_quality_control=True,
                    is_his3delta_control=True,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="2",
                protein=proteomics.Protein(id="protein_b"),
                condition=omics.GeneKnockout(
                    standard_name="standard_name_b",
                    systematic_name="systematic_name_b",
                    expression_level=2.0,
                ),
                abundance_level=2.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=2,
                    well_nr=2,
                    batch_nr="hrp2",
                    is_quality_control=True,
                    is_his3delta_control=True,
                ),
            ),
        ]
        return result

    @pytest.fixture
    def proteomics_data(self, data):
        return proteomics.Proteomics(data=data)

    def test_return_as_list(self, proteomics_data, data):
        observed = proteomics_data.return_as_list()
        assert observed == data

    def test_group_by_id(self, proteomics_data, data):
        observed = proteomics_data.group_by_id()
        assert len(observed) == 2
        assert len(observed["protein_a"]) == 1
        assert len(observed["protein_b"]) == 1
        assert observed["protein_a"][0] == data[0]
        assert observed["protein_b"][0] == data[1]

    def test_group_by_condition(self, proteomics_data, data):
        observed = proteomics_data.group_by_condition()
        assert len(observed) == 2
        assert len(observed["standard_name_a"]) == 1
        assert len(observed["standard_name_b"]) == 1
        assert observed["standard_name_a"][0] == data[0]
        assert observed["standard_name_b"][0] == data[1]

    def test_get_by_id(self, proteomics_data, data):
        observed = proteomics_data.get_by_id("1")
        assert observed == data[0]
        with pytest.raises(ValueError):
            proteomics_data.get_by_id("3")

    def test_set(self, proteomics_data, data):
        new_data = [
            proteomics.ProteinAbundanceProfile(
                id="3",
                protein=proteomics.Protein(id="protein_c"),
                condition=omics.GeneKnockout(
                    standard_name="standard_name_c",
                    systematic_name="systematic_name_c",
                    expression_level=3.0,
                ),
                abundance_level=3.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=3,
                    well_nr=3,
                    batch_nr="hrp3",
                    is_quality_control=True,
                    is_his3delta_control=True,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="4",
                protein=proteomics.Protein(id="protein_d"),
                condition=omics.GeneKnockout(
                    standard_name="standard_name_d",
                    systematic_name="systematic_name_d",
                    expression_level=4.0,
                ),
                abundance_level=4.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=4,
                    well_nr=4,
                    batch_nr="hrp4",
                    is_quality_control=True,
                    is_his3delta_control=True,
                ),
            ),
        ]
        observed = proteomics_data.set(new_data)
        assert observed.return_as_list() == new_data
