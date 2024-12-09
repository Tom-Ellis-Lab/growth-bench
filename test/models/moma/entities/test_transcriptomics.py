import pytest

from bench.models.moma.entities import omics, transcriptomics


class TestTranscript:

    def test_init(self):
        observed = transcriptomics.Transcript(id="transcript_id")
        assert observed.id == "transcript_id"


class TestTranscriptomicsProfileData:

    def test_init(self):
        observed = transcriptomics.TranscriptProfileData(
            transcript_id="transcript_id",
            transcript_expression_level=1.0,
            ko_gene_standard_name="gene_standard_name",
        )
        assert observed.transcript_id == "transcript_id"
        assert observed.transcript_expression_level == 1.0
        assert observed.ko_gene_standard_name == "gene_standard_name"


class TestTranscriptExpressionProfile:

    def test_init(self):
        observed = transcriptomics.TranscriptExpressionProfile(
            id="1",
            transcript=transcriptomics.Transcript(id="transcript_id"),
            condition=omics.GeneKnockout(
                standard_name="gene_standard_name",
                systematic_name="gene_systematic_name",
                expression_level=1.0,
            ),
            expression_level=1.0,
        )
        assert observed.id == "1"
        assert observed.molecular_entity.id == "transcript_id"
        assert observed.value == 1.0
        assert observed.condition.name == "gene_standard_name"


class TestTranscriptomics:

    @pytest.fixture
    def data(self):
        result = [
            transcriptomics.TranscriptExpressionProfile(
                id="1",
                transcript=transcriptomics.Transcript(id="transcript_a"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_a",
                    systematic_name="gene_systematic_name_a",
                    expression_level=1.0,
                ),
                expression_level=1.0,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="2",
                transcript=transcriptomics.Transcript(id="transcript_b"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_b",
                    systematic_name="gene_systematic_name_b",
                    expression_level=2.0,
                ),
                expression_level=2.0,
            ),
        ]

        return result

    @pytest.fixture
    def transcriptomics_data(self, data):
        return transcriptomics.Transcriptomics(data=data)

    def test_return_as_list(self, data, transcriptomics_data):
        observed = transcriptomics_data.return_as_list()
        assert observed == data

    def test_get_by_id(self, data, transcriptomics_data):
        observed = transcriptomics_data.get_by_id("1")
        assert observed == data[0]

        with pytest.raises(ValueError):
            transcriptomics_data.get_by_id("3")

    def test_group_by_id(self, data, transcriptomics_data):

        observed = transcriptomics_data.group_by_id()

        assert len(observed) == 2
        assert "transcript_a" in observed
        assert "transcript_b" in observed

        assert len(observed["transcript_a"]) == 1
        assert len(observed["transcript_b"]) == 1

        assert observed["transcript_a"][0] == data[0]
        assert observed["transcript_b"][0] == data[1]

    def test_group_by_condition(self, data, transcriptomics_data):
        observed = transcriptomics_data.group_by_condition()

        assert len(observed) == 2
        assert "gene_standard_name_a" in observed
        assert "gene_standard_name_b" in observed

        assert len(observed["gene_standard_name_a"]) == 1
        assert len(observed["gene_standard_name_b"]) == 1

        assert observed["gene_standard_name_a"][0] == data[0]
        assert observed["gene_standard_name_b"][0] == data[1]

    def test_set(self, transcriptomics_data):
        new_data = [
            transcriptomics.TranscriptExpressionProfile(
                id="3",
                transcript=transcriptomics.Transcript(id="transcript_c"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_c",
                    systematic_name="gene_systematic_name_c",
                    expression_level=3.0,
                ),
                expression_level=3.0,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="4",
                transcript=transcriptomics.Transcript(id="transcript_d"),
                condition=omics.GeneKnockout(
                    standard_name="gene_standard_name_d",
                    systematic_name="gene_systematic_name_d",
                    expression_level=4.0,
                ),
                expression_level=4.0,
            ),
        ]
        observed = transcriptomics_data.set(new_data)
        assert observed._data == new_data
