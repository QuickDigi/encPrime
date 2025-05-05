const DeepLearn = require('./secret/DeepLearn.js');
const fs = require('fs');

class Tensor {
    #net;
    #saveFilePath = `./model-lst${new Date().getDay()}.json`;

    constructor
        ({
            inputSize = 144,
            outputSize = 144,
            hiddenNodes = [3],
            ActiveMode = 'sigmoid',
            ReinforcementLearning = false,
        }) {
        this.#net = new DeepLearn({
            inputSize,
            outputSize,
            hiddenLayers: hiddenNodes,
            activation: ActiveMode,
            ReinforcementLearning
        });
    }

    /**
     * Trains the neural network using the provided training data and configuration options.
     * 
     * @param {Array} trainingData - An array of training examples, each containing input and output data.
     * @param {Object} options - Configuration options for the training process.
     * @param {number} [options.epochs=5000] - The maximum number of training iterations.
     * @param {number} [options.errorThresh=0.005] - The error threshold to stop training.
     * @param {boolean} [options.log=true] - Whether to log training progress.
     * @param {number} [options.logPeriod=10] - The number of iterations between logging updates.
     * @param {number} [options.learningRate=0.3] - The learning rate for the training algorithm.
     * @param {number} [options.momentum=0.1] - The momentum for the training algorithm.
     * @param {Function|null} [options.callback=null] - A callback function to be called during training.
     * @param {number} [options.callbackPeriod=10] - The number of iterations between callback invocations.
     * @param {number} [options.timeout=Infinity] - The maximum time (in milliseconds) to train for.
     * @param {number} [options.dropout=0] - The dropout rate to apply during training.
     * @param {string} [options.activation='sigmoid'] - The activation function to use.
     * @param {number} [options.leakyReluAlpha=0.01] - The alpha value for the leaky ReLU activation function.
     * @returns {Object} An object containing the training results, including error and number of iterations.
     */
    train(trainingData, {
        epochs = 5000,
        errorThresh = 0.005,
        log = true,
        logPeriod = 10,
        learningRate = 0.3,
        momentum = 0.1,
        callback = null,
        callbackPeriod = 10,
        timeout = Infinity,
        dropout = 0,
        activation = 'sigmoid',
        leakyReluAlpha = 0.01,
    } = {}) {
        console.log(`🚀 Starting training with ${trainingData.length} examples...`);
        console.log(`⚙️ Configuration: epochs=${epochs}, learningRate=${learningRate}, activation=${activation}`);
    
        // Validate input/output sizes before training
        if (trainingData.length > 0) {
            const sampleInput = trainingData[0].input;
            const sampleOutput = trainingData[0].output;
            console.log(`📊 Sample input size: ${sampleInput.length}, Sample output size: ${sampleOutput.length}`);
        }
    
        const startTime = Date.now();
        let result;
    
        try {
            result = this.#net.train(trainingData, {
                iterations: epochs,
                errorThresh,
                log: false,
                logPeriod,
                learningRate,
                momentum,
                callback: callback || ((stats) => {
                    { log && console.log(`📈 Epoch ${stats.iterations}: error=${stats.error.toFixed(20)}`) };
                }),
                callbackPeriod,
                timeout,
                dropout,
                activation,
                leakyReluAlpha,
            });
    
            const trainingTime = (Date.now() - startTime) / 1000;
            console.log(`✅ Training completed in ${trainingTime.toFixed(2)} seconds with error: ${result.error.toFixed(5)}`);
    
        } catch (error) {
            console.error(`🔥 Training error: ${error.message}`);
    
            // Check for input size mismatch
            if (error.message.includes('input length') && error.message.includes('must match options.inputSize')) {
                const actualSize = parseInt(error.message.match(/input length (\d+)/)[1]);
                const expectedSize = parseInt(error.message.match(/options.inputSize of (\d+)/)[1]);
    
                console.error(`⚠️ Input size mismatch: Got ${actualSize}, expected ${expectedSize}`);
                console.error(`💡 Tip: Update your constructor to match the input size: inputSize=${actualSize}`);
    
                // Auto-fix by recreating the network with correct size
                console.log(`🔄 Automatically recreating network with inputSize=${actualSize}...`);
                this.#net = new DeepLearn({
                    inputSize: actualSize,
                    outputSize: actualSize, // Assuming same size for input/output
                    hiddenLayers: [Math.ceil(actualSize / 2), Math.ceil(actualSize / 2), Math.ceil(actualSize / 3)],
                    activation,
                    ReinforcementLearning: false
                });
    
                console.log(`🔄 Network recreated with inputSize=${actualSize}`);
                return this.train(trainingData, {
                    epochs, errorThresh, log, logPeriod, learningRate,
                    momentum, callback, callbackPeriod, timeout, dropout,
                    activation, leakyReluAlpha, beta1, beta2
                }); // Retry with same params
            }
    
            return { error: 1, iterations: 0 }; // Return failed result
        }
    
        return result;
    }

    /**
     * Saves the current model to a file.
     * 
     * This function serializes the neural network model to JSON format and saves it to a file.
     * If no file path is provided, it uses the default save file path.
     * 
     * @param {string} [filePath=this.#saveFilePath] - The path where the model will be saved.
     *                                                 If not provided, the default path is used.
     * @returns {void}
     */
    saveModel(filePath = this.#saveFilePath) {
        const json = this.#net.SaveModel();
        fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
        console.log(`✅ Model saved to ${filePath}`);
    }

    /**
     * Loads a previously saved model from a file.
     * 
     * This function reads a JSON file containing a serialized neural network model
     * and loads it into the current instance of the neural network.
     * 
     * @param {string} [filePath=this.#saveFilePath] - The path to the file containing the saved model.
     *                                                 If not provided, it uses the default save file path.
     * @returns {void}
     * @throws {Error} If there's an error reading the file or parsing the JSON.
     */
    loadModel(filePath = this.#saveFilePath) {
        try {
            const data = fs.readFileSync(filePath, 'utf8');
            this.#net.LoadModel(JSON.parse(data));
            console.log(`✅ Model loaded from ${filePath}`);
        } catch (error) {
            console.error(`❌ Failed to load model from ${filePath}:`, error);
        }
    }

    /**
     * Loads a model from a specified URL.
     * 
     * This function attempts to load a neural network model from a given URL.
     * It first checks if a URL is provided, and if not, it logs an error.
     * 
     * @param {string} url - The URL from which to load the model.
     * @returns {Promise<void>|undefined} A promise that resolves when the model is loaded successfully,
     *                                    or undefined if the URL is not provided.
     * @throws {Error} Potentially throws an error if the model loading fails.
     */
    loadModelURL(url) {
        if (!url) {
            console.error('❌ URL is required for loadModelURL');
            return;
        }
    
        return this.#net.loadModelURL(url);
    }

    /**
     * Performs a prediction using the trained neural network.
     * 
     * @param {Array|number} input - The input data for prediction. This should match the format and size
     *                               expected by the neural network.
     * @returns {Array|number|null} The prediction result from the neural network. Returns null if the
     *                              prediction fails due to an error.
     */
    predict(input) {
        try {
            return this.#net.run(input);
        } catch (error) {
            console.error('❌ Prediction failed:', error);
            return null;
        }
    }
}

class NanoTensor {
    static BITS_PER_CHAR = 16;
    static MAX_CHARS = 16;
    static TOTAL_BITS = NanoTensor.BITS_PER_CHAR * NanoTensor.MAX_CHARS;

    /**
     * يحول نص لـ tensor (مصفوفة 0/1)
     * @param {string} text
     * @param {number} bits
     * @returns {number[]}
     */
    static textToTensor(text, bits = NanoTensor.TOTAL_BITS) {
        const safeText = text.padEnd(NanoTensor.MAX_CHARS).slice(0, NanoTensor.MAX_CHARS);

        let binary = Array.from(safeText)
            .map((c) =>
                c
                    .charCodeAt(0) // 0–65535
                    .toString(2) // to binary
                    .padStart(NanoTensor.BITS_PER_CHAR, "0")
            )
            .join("");

        binary = binary.padEnd(bits, "0").slice(0, bits);

        return binary.split("").map((b) => Number(b));
    }

    /**
     * يحول tensor (0/1) لنص
     * @param {number[]} tensor
     * @param {number} bits
     * @returns {string}
     */
    static tensorToText(tensor, bits = NanoTensor.TOTAL_BITS) {
        const bin = tensor
            .map((n) => (Number(n) > 0.5 ? "1" : "0"))
            .join("")
            .padEnd(bits, "0")
            .slice(0, bits);

        const chars = bin.match(new RegExp(`.{1,${NanoTensor.BITS_PER_CHAR}}`, "g")) || [];

        return chars
            .map((b) => String.fromCharCode(parseInt(b, 2)))
            .join("")
            .trim();
    }

    /**
     * يحول مصفوفة أرقام لـ tensor ثنائي
     * @param {Array<number>} array
     * @param {number} bits
     * @returns {number[]}
     */
    static arrayToTensor(array, bits = NanoTensor.TOTAL_BITS) {
        const binaryArray = array.map(value =>
            Math.round(Number(value))
                .toString(2)
                .padStart(NanoTensor.BITS_PER_CHAR, "0")
                .slice(-NanoTensor.BITS_PER_CHAR)
        );
        let binary = binaryArray.join("");
        binary = binary.padEnd(bits, "0").slice(0, bits);
        return {
            tensor: binary.split("").map(b => Number(b)),
            length: array.length
        };
    }


    /**
     * Converts a binary tensor representation back into an array of numbers.
     * 
     * @param {number[]} tensor - The binary tensor to convert, represented as an array of numbers (0 or 1).
     * @param {number} [bits=NanoTensor.TOTAL_BITS] - The total number of bits in the tensor. Defaults to NanoTensor.TOTAL_BITS.
     * @returns {number[]} An array of numbers decoded from the binary tensor.
     */
    static tensorToArray({ tensor, length }, bits = NanoTensor.TOTAL_BITS) {
        const bin = tensor
            .map(n => (Number(n) > 0.5 ? "1" : "0"))
            .join("")
            .padEnd(bits, "0")
            .slice(0, bits);

        const chunks = bin.match(new RegExp(`.{1,${NanoTensor.BITS_PER_CHAR}}`, "g")) || [];
        // هيرجع بس أول 'length' عناصر بدل كل اللي في الـ chunks
        return chunks.slice(0, length).map(chunk => parseInt(chunk, 2));
    }
}


module.exports = { Tensor, NanoTensor };