import pytest

from bench.models.moma.gateways import proteomics_gateway
from bench.models.moma.entities import proteomics


class TestProteomicsGateway:

    @pytest.fixture
    def csv_content(self):
        result = "Protein.Group,10_9_hpr1_ko_YAL059W_ECM1_0.47,1_0_hpr11_qc_qc_qc.1_NA,12_11_hpr14_HIS3_YOR202W_HIS3.1_0.38\nA5Z2X5,1.1,4.4,7.7\n"
        return result

    @pytest.fixture
    def file_path(self, tmp_path):
        result = tmp_path / "test.csv"
        return result

    @pytest.fixture
    def expected(self):
        result = [
            proteomics.ProteinProfileData(
                protein_id="A5Z2X5",
                protein_abundance_level=1.1,
                injection_nr=10,
                well_nr=9,
                batch_nr="hpr1",
                is_quality_control=False,
                is_his3delta_control=False,
                ko_gene_systematic_name="YAL059W",
                ko_gene_standard_name="ECM1",
                ko_gene_expression_level=0.47,
            ),
            proteomics.ProteinProfileData(
                protein_id="A5Z2X5",
                protein_abundance_level=4.4,
                injection_nr=1,
                well_nr=0,
                batch_nr="hpr11",
                is_quality_control=True,
                is_his3delta_control=False,
                ko_gene_systematic_name="qc",
                ko_gene_standard_name="qc.1",
                ko_gene_expression_level=None,
            ),
            proteomics.ProteinProfileData(
                protein_id="A5Z2X5",
                protein_abundance_level=7.7,
                injection_nr=12,
                well_nr=11,
                batch_nr="hpr14",
                is_quality_control=False,
                is_his3delta_control=True,
                ko_gene_systematic_name="YOR202W",
                ko_gene_standard_name="HIS3.1",
                ko_gene_expression_level=0.38,
            ),
        ]

        return result

    def test_get(self, csv_content, file_path, expected):
        file_path.write_text(csv_content)
        observed = proteomics_gateway.ProteomicsGateway(file_path=file_path).get()
        assert observed == expected

    @pytest.fixture
    def wrong_header_csv_content(self):
        result = "wrong_header,YAL001C,YBL002W\nA5Z2X5,1.1,4.4\nD6VTK4,2.2,5.5\nO13297,3.3,6.6\n"
        return result

    def test_get_wrong_header(self, wrong_header_csv_content, file_path):
        file_path.write_text(wrong_header_csv_content)
        with pytest.raises(ValueError):
            proteomics_gateway.ProteomicsGateway(file_path=file_path).get()
