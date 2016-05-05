

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;


public class NeuralNetworks {

	private static Logger log = LoggerFactory.getLogger(NeuralNetworks.class);

	public static void main(String[] args) throws Exception {
		final int numRows = 28;
		final int numColumns = 28;
		int outputNum = 10;
		int numSamples = 60000;
		int batchSize = 100;
		int iterations = 10;
		int seed = 123;
		int listenerFreq = batchSize / 5;

		log.info("Load data....");
		DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples,
				true);

		log.info("Build model....");
		 MultiLayerNetwork model = softMaxRegression(seed, iterations, numRows, numColumns, outputNum);
//		// MultiLayerNetwork model = deepBeliefNetwork(seed, iterations,
//		// numRows, numColumns, outputNum);
//		MultiLayerNetwork model = deepConvNetwork(seed, iterations, numRows,
//				numColumns, outputNum);

		model.init();
		model.setListeners(Collections
				.singletonList((IterationListener) new ScoreIterationListener(
						listenerFreq)));

		log.info("Train model....");
		model.fit(iter); // achieves end to end pre-training

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputNum);

		DataSetIterator testIter = new MnistDataSetIterator(100, 10000);
		while (testIter.hasNext()) {
			DataSet testMnist = testIter.next();
			INDArray predict2 = model.output(testMnist.getFeatureMatrix());
			eval.eval(testMnist.getLabels(), predict2);
		}

		log.info(eval.stats());
		log.info("****************Example finished********************");

	}

	private static MultiLayerNetwork softMaxRegression(int seed,
			int iterations, int numRows, int numColumns, int outputNum) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.gradientNormalization(
						GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0)
				.iterations(iterations)
				.momentum(0.5)
				.momentumAfter(Collections.singletonMap(3, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.list(1)
				.layer(0,
						new OutputLayer.Builder(
								LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation("softmax")
								.nIn(numColumns * numRows).nOut(outputNum)
								.build()).pretrain(true).backprop(false)
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		return model;
	}

	private static MultiLayerNetwork deepBeliefNetwork(int seed,
			int iterations, int numRows, int numColumns, int outputNum) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.gradientNormalization(
						GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0)
				.iterations(iterations)
				.momentum(0.5)
				.momentumAfter(Collections.singletonMap(3, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				.list(4)
				.layer(0,
						new RBM.Builder().nIn(numRows * numColumns).nOut(500)
								.weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunction.RMSE_XENT)
								.visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build())
				.layer(1,
						new RBM.Builder().nIn(500).nOut(250)
								.weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunction.RMSE_XENT)
								.visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build())
				.layer(2,
						new RBM.Builder().nIn(250).nOut(200)
								.weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunction.RMSE_XENT)
								.visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build())
				.layer(3,
						new OutputLayer.Builder(
								LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation("softmax").nIn(200).nOut(outputNum)
								.build()).pretrain(true).backprop(false)
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		return model;
	}

	private static MultiLayerNetwork deepConvNetwork(int seed, int iterations,
			int numRows, int numColumns, int outputNum) {
		MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.activation("sigmoid")
				.weightInit(WeightInit.DISTRIBUTION)
				.dist(new NormalDistribution(0.0, 0.01))
				// .learningRate(7*10e-5)
				.learningRate(1e-3)
				.learningRateScoreBasedDecayRate(1e-1)
				.optimizationAlgo(
						OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list(7)
				.layer(0,
						new ConvolutionLayer.Builder(new int[] { 5, 5 },
								new int[] { 1, 1 }).name("cnn1")
								.nIn(numRows * numColumns).nOut(6).build())
				.layer(1,
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX, new int[] {
										2, 2 }, new int[] { 2, 2 }).name(
								"maxpool1").build())
				.layer(2,
						new ConvolutionLayer.Builder(new int[] { 5, 5 },
								new int[] { 1, 1 }).name("cnn2").nOut(16)
								.biasInit(1).build())
				.layer(3,
						new SubsamplingLayer.Builder(
								SubsamplingLayer.PoolingType.MAX, new int[] {
										2, 2 }, new int[] { 2, 2 }).name(
								"maxpool2").build())
				.layer(4,
						new DenseLayer.Builder().name("ffn1").nOut(120).build())
				.layer(5,
						new DenseLayer.Builder().name("ffn2").nOut(84).build())
				.layer(6,
						new OutputLayer.Builder(
								LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.name("output").nOut(outputNum)
								.activation("softmax") // radial basis function
														// required
								.build()).backprop(true).pretrain(false)
				.cnnInputSize(numRows, numColumns, 1);

		MultiLayerNetwork model = new MultiLayerNetwork(conf.build());

		return model;
	}

}