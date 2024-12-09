import pytest

from bench.models.moma.filter_utils import omics_filter_utils
from bench.models.moma.entities import proteomics, omics, transcriptomics, fluxomics


class TestRemoveQualityControl:

    @pytest.fixture
    def input(self):
        result = [
            proteomics.ProteinAbundanceProfile(
                id="1",
                protein=proteomics.Protein(id="A5Z2X5"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                abundance_level=0.5,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr3",
                    is_quality_control=False,
                    is_his3delta_control=False,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="2",
                protein=proteomics.Protein(id="D6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                abundance_level=0.6,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=True,
                    is_his3delta_control=False,
                ),
            ),
        ]
        return result

    @pytest.fixture
    def omics_data(self, input):
        return proteomics.Proteomics(data=input)

    def test_remove_quality_control(self, omics_data, input):
        strategy = omics_filter_utils.RemoveQualityControl().get_strategy()
        observed = strategy(omics_data)
        expected = proteomics.Proteomics(data=[input[0]])
        assert observed.return_as_list() == expected.return_as_list()


class TestRemoveHis3deltaControl:

    @pytest.fixture
    def input(self):
        result = [
            proteomics.ProteinAbundanceProfile(
                id="1",
                protein=proteomics.Protein(id="A5Z2X5"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                abundance_level=0.5,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr3",
                    is_quality_control=False,
                    is_his3delta_control=False,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="2",
                protein=proteomics.Protein(id="D6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                abundance_level=0.6,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=True,
                    is_his3delta_control=True,
                ),
            ),
        ]
        return result

    @pytest.fixture
    def omics_data(self, input):
        return proteomics.Proteomics(data=input)

    def test_remove_his3delta_control(self, omics_data, input):
        strategy = omics_filter_utils.RemoveHis3deltaControl().get_strategy()
        observed = strategy(omics_data)
        expected = proteomics.Proteomics(data=[input[0]])
        assert observed.return_as_list() == expected.return_as_list()


class TestRemoveZeroSamples:

    @pytest.fixture
    def data(self):
        result = [
            transcriptomics.TranscriptExpressionProfile(
                id="1",
                transcript=transcriptomics.Transcript(id="YAL001C"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                expression_level=0.5,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="2",
                transcript=transcriptomics.Transcript(id="YAL002W"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                expression_level=0.0,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="3",
                transcript=transcriptomics.Transcript(id="YAL002W"),
                condition=omics.GeneKnockout(standard_name="TF5D"),
                expression_level=0.0,
            ),
        ]
        return result

    @pytest.fixture
    def omics_data(self, data):
        return transcriptomics.Transcriptomics(data=data)

    def test_remove_zero_samples(self, omics_data, data):
        strategy = omics_filter_utils.RemoveZeroSamples().get_strategy()
        observed = strategy(omics_data)
        expected = transcriptomics.Transcriptomics(data=[data[0]])
        assert observed.return_as_list() == expected.return_as_list()


class TestFilter:

    @pytest.fixture
    def strategies(self):
        return [
            omics_filter_utils.RemoveQualityControl(),
            omics_filter_utils.RemoveHis3deltaControl(),
            omics_filter_utils.RemoveZeroSamples(),
        ]

    @pytest.fixture
    def data(self):
        result = [
            proteomics.ProteinAbundanceProfile(
                id="1",
                protein=proteomics.Protein(id="A5Z2X5"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                abundance_level=0.5,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr3",
                    is_quality_control=False,
                    is_his3delta_control=False,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="2",
                protein=proteomics.Protein(id="D6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                abundance_level=0.6,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=True,
                    is_his3delta_control=False,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="3",
                protein=proteomics.Protein(id="C6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC5"),
                abundance_level=0.45,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=False,
                    is_his3delta_control=True,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="4",
                protein=proteomics.Protein(id="E6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC6"),
                abundance_level=0.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=False,
                    is_his3delta_control=False,
                ),
            ),
            proteomics.ProteinAbundanceProfile(
                id="5",
                protein=proteomics.Protein(id="E6VTK4"),
                condition=omics.GeneKnockout(standard_name="TFC7"),
                abundance_level=0.0,
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=1,
                    well_nr=2,
                    batch_nr="hpr4",
                    is_quality_control=False,
                    is_his3delta_control=False,
                ),
            ),
        ]
        return result

    @pytest.fixture
    def omics_data(self, data):
        return proteomics.Proteomics(data=data)

    def test_filter(self, omics_data, strategies, data):
        filter = omics_filter_utils.Filter(strategies=strategies)
        observed = filter.filter(omics_data)
        expected = proteomics.Proteomics(data=[data[0]])
        assert observed.return_as_list() == expected.return_as_list()


class TestWrongFilter:

    @pytest.fixture
    def strategies(self):
        return [
            omics_filter_utils.RemoveQualityControl(),
            omics_filter_utils.RemoveHis3deltaControl(),
        ]

    @pytest.fixture
    def fluxomics_data(self):
        result = [
            fluxomics.MetabolicFLuxProfile(
                id="1",
                reaction=fluxomics.MetabolicReaction(id="R1"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                flux_rate=0.5,
            ),
            fluxomics.MetabolicFLuxProfile(
                id="2",
                reaction=fluxomics.MetabolicReaction(id="R2"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                flux_rate=0.4,
            ),
            fluxomics.MetabolicFLuxProfile(
                id="3",
                reaction=fluxomics.MetabolicReaction(id="R3"),
                condition=omics.GeneKnockout(standard_name="TFC5"),
                flux_rate=0.6,
            ),
        ]
        return result

    @pytest.fixture
    def fluxomics_object(self, fluxomics_data):
        return fluxomics.Fluxomics(data=fluxomics_data)

    def test_filter_fluxomics(self, fluxomics_object, strategies):
        filter = omics_filter_utils.Filter(strategies=strategies)
        with pytest.raises(NotImplementedError):
            observed = filter.filter(fluxomics_object)

    @pytest.fixture
    def transcriptomics_data(self):
        result = [
            transcriptomics.TranscriptExpressionProfile(
                id="1",
                transcript=transcriptomics.Transcript(id="YAL001C"),
                condition=omics.GeneKnockout(standard_name="TFC3"),
                expression_level=0.5,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="2",
                transcript=transcriptomics.Transcript(id="YAL002W"),
                condition=omics.GeneKnockout(standard_name="TFC4"),
                expression_level=0.4,
            ),
            transcriptomics.TranscriptExpressionProfile(
                id="3",
                transcript=transcriptomics.Transcript(id="YAL002W"),
                condition=omics.GeneKnockout(standard_name="TF5D"),
                expression_level=0.6,
            ),
        ]
        return result

    @pytest.fixture
    def transcriptomics_object(self, transcriptomics_data):
        return transcriptomics.Transcriptomics(data=transcriptomics_data)

    def test_filter_transcriptomics(self, transcriptomics_object, strategies):
        filter = omics_filter_utils.Filter(strategies=strategies)
        with pytest.raises(NotImplementedError):
            observed = filter.filter(transcriptomics_object)


class TestFilterFactory:

    @pytest.fixture
    def strategies(self):
        return [
            omics_filter_utils.RemoveQualityControl(),
            omics_filter_utils.RemoveHis3deltaControl(),
            omics_filter_utils.RemoveZeroSamples(),
        ]

    @pytest.fixture
    def factory(self, strategies):
        return omics_filter_utils.FilterFactory(strategies=strategies)

    @pytest.fixture
    def expected_filter(self, strategies):
        return omics_filter_utils.Filter(strategies=strategies)

    def test_create_filter(self, factory, expected_filter):
        observed = factory.create_filter()
        assert observed._strategies == expected_filter._strategies
