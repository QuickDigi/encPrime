const brain = require('brain.js');
const fs = require('fs');

class Tensor {
    #net;
    #saveFilePath = `./model-lst${new Date().getDay()}.enpt`;

    constructor({ inputSize = 144, outputSize = 144, hiddenLayers = [50, 50, 50] }) {
        this.#net = new brain.NeuralNetwork({
            inputSize,
            outputSize,
            hiddenLayers,
        });
    }

    train(trainingData, { epochs = 10000, errorThresh = 0.002, log = false, logPeriod = 100, learningRate = 0.005 }) {
        const inputSize = trainingData[0].input.length;
        const outputSize = trainingData[0].output.length;

        this.#net.train(trainingData, {
            iterations: epochs,
            errorThresh,
            log,
            logPeriod,
            learningRate,
            callback: (status) => {
                console.log(`${status.iterations}/${epochs} error: ${status.error}`)
            },
            onFinish: (weights) => {
                console.log('�� Training completed!');
            },
            onIteration: (iteration, error) => {
                console.log(`Iteration ${iteration}: Error = ${error}`);
            },
            onEpoch: (epoch, error) => {
                console.log(`Epoch ${epoch}: Error = ${error}`);
            },
            onProgress: (progress) => {
                console.log(`Progress: ${progress}`);
            },
            onSync: (data) => {
                console.log(`Sync: ${data}`);
            },
            onWeightsChange: (deltaWeights, prevDeltaWeights) => {
                console.log(`Weights change: Delta = ${deltaWeights}, Prev Delta = ${prevDeltaWeights}`);
            },
            onWeightsSave: (data) => {
                console.log(`Weights saved: ${data}`);
            },
            onWeightsLoad: (data) => {
                console.log(`Weights loaded: ${data}`);
            },
            onSave: (data) => {
                console.log(`Model saved: ${data}`);
            },
            onLoad: (data) => {
                console.log(`Model loaded: ${data}`);
            },
            onReset: (data) => {
                console.log(`Model reset: ${data}`);
            },
            onWeightsRandomize: (data) => {
                console.log(`Weights randomized: ${data}`);
            },

        });
    }

    saveModel(filePath = this.#saveFilePath) {
        const json = this.#net.toJSON();
        fs.writeFileSync(filePath, JSON.stringify(json, null, 2));
        console.log(`✅ Model saved to ${filePath}`);
    }

    loadModel(filePath = this.#saveFilePath) {
        try {
            const data = fs.readFileSync(filePath, 'utf8');
            this.#net.fromJSON(JSON.parse(data));
            console.log(`✅ Model loaded from ${filePath}`);
        } catch (error) {
            console.error(`❌ Failed to load model from ${filePath}:`, error);
        }
    }

    predict(input) {
        try {
            return this.#net.run(input);
        } catch (error) {
            console.error('❌ Prediction failed:', error);
            return null;
        }
    }
}

const BITS_PER_CHAR = 8;
const MAX_CHARS = 16;
const TOTAL_BITS = BITS_PER_CHAR * MAX_CHARS;

const NanoTensor = {
    TextToTensor: (text, bits = TOTAL_BITS) => {
        // 1. نحافظ على الطول
        const safeText = text.padEnd(MAX_CHARS).slice(0, MAX_CHARS);

        // 2. نحول كل حرف إلى binary ثابت الطول
        let binary = safeText
            .split("")
            .map(c => {
                const code = c.charCodeAt(0);
                return code.toString(2).padStart(BITS_PER_CHAR, "0");
            })
            .join("");

        // 3. نكمل بالباقي أصفار لو ناقص
        binary = binary.padEnd(bits, "0").slice(0, bits);

        // 4. نحولها لأرقام (0 أو 1)
        return binary.split("").map(b => Number(b));
    },

    TensorToText: (tensor, bits = TOTAL_BITS) => {
        // 1. تأكد إنها أرقام 0 أو 1
        const bin = tensor
            .map(n => (Number(n) > 0.5 ? "1" : "0"))
            .join("")
            .padEnd(bits, "0")
            .slice(0, bits);

        // 2. نقسمها إلى 8 بت لكل حرف
        const chars = bin.match(new RegExp(`.{1,${BITS_PER_CHAR}}`, "g")) || [];

        // 3. نحول كل 8 بت إلى حرف
        return chars
            .map(b => String.fromCharCode(parseInt(b, 2)))
            .join("")
            .trim();
    }
};

module.exports = { Tensor, NanoTensor };