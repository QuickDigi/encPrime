const brain = require('brain.js');
const math = require('mathjs');

class DeepLearn {
    constructor(options = {}) {
        const isReinforcement = !!options.ReinforcementLearning;

        // ✅ تضخيم الطبقات قبل بناء الشبكة
        if (isReinforcement) {
            console.log('📝 Note: Reinforcement learning requires more than 4GB of RAM.');
            // options.activation = 'relu';
            // لو فيه hiddenLayers معرف، نضاعفه
            if (Array.isArray(options.hiddenLayers)) {
                options.hiddenLayers = options.hiddenLayers.map(l => l * 4);
            }

            // تعديل learningRate قبل الإنشاء
            options.learningRate = (options.learningRate || 0.001) * 4;
        }

        // 🧠 بناء الشبكة العصبية بعد تعديل options
        this.net = new brain.NeuralNetwork(options);
        this.m = math.zeros(this.net.weights.length);
        this.v = math.zeros(this.net.weights.length);
        this.t = 0; // عداد الخطوات للمحسن

        // باقي المتغيرات
        this.optimizer = options.optimizer || 'adam';
        this.learningRate = options.learningRate || 0.001;
        this.batchSize = options.batchSize || 32;
        this.qTable = new Map();
        this.isReinforcement = isReinforcement;

        // RL specific parameters
        this.gamma = options.gamma || 0.95; // discount factor
        this.epsilon = options.epsilon || 1.0; // exploration rate
        this.epsilonMin = options.epsilonMin || 0.01;
        this.epsilonDecay = options.epsilonDecay || 0.995;
        this.memory = [];
        this.maxMemory = options.maxMemory || 1000;

    }

    train(data, options) {
        const batchedData = this.#batchData(data, this.batchSize);
        let totalError = 0;
        delete options.ReinforcementLearning;
        for (let epoch = 0; epoch < options.epochs; epoch++) {
            for (const batch of batchedData) {
                const gradients = this.#computeGradients(batch);
                this.#updateWeights(gradients);
            }

            if (options.log && epoch % options.logPeriod === 0) {
                const error = this.#validateError(data);
                totalError += error;
                console.log(`Epoch ${epoch}: error = ${error}`);
            }

            if (totalError / options.logPeriod < options.errorThresh) {
                console.log('Early stopping due to low error');
                break;
            }
        }

        return this.net.train(
            data,
            options
        );
    }

    run(input) {
        return this.net.run(input);
    }

    predict(input) {
        const output = this.run(input);
        return this.#argmax(output);
    }

    SaveModel() {
        return this.net.toJSON();
    }

    LoadModel(json) {
        this.net.fromJSON(json);
    }

    loadModelURL(url) {
        return new Promise((resolve, reject) => {
            // Special handling for filemail.com URLs
            if (url.includes('filemail.com')) {
                console.log('Detected filemail.com URL, using direct download method');
                return this.#downloadFilemailModel(url, resolve, reject);
            }

            // Try multiple CORS proxies in case one fails
            const proxies = [
                '', // Try direct first
                'https://cors-anywhere.herokuapp.com/',
                'https://api.allorigins.win/raw?url='
            ];

            // Add custom headers that might help with authorization
            const headers = {
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            };

            // Try each proxy in sequence
            const tryNextProxy = (index) => {
                if (index >= proxies.length) {
                    return reject(new Error('All proxies failed to load the model'));
                }

                const proxyUrl = proxies[index] + url;
                console.log(`Attempting to load model from: ${proxyUrl}`);

                fetch(proxyUrl, { headers })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Network response was not ok: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(json => {
                        this.LoadModel(json);
                        console.log(`✅ Model loaded from URL: ${url} using proxy: ${proxies[index]}`);
                        resolve();
                    })
                    .catch(error => {
                        console.warn(`Failed with proxy ${proxies[index]}: ${error.message}`);
                        // Try next proxy
                        tryNextProxy(index + 1);
                    });
            };

            // Start with the first proxy
            tryNextProxy(0);
        });
    }

    // Special method to handle filemail.com downloads
    #downloadFilemailModel(url, resolve, reject) {
        const https = require('https');
        const http = require('http');
        const { URL } = require('url');

        try {
            const parsedUrl = new URL(url);
            const protocol = parsedUrl.protocol === 'https:' ? https : http;

            const options = {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/json'
                }
            };

            console.log(`Attempting direct download from: ${url}`);

            const req = protocol.get(url, options, (res) => {
                if (res.statusCode !== 200) {
                    reject(new Error(`Failed to download: Status code ${res.statusCode}`));
                    return;
                }

                let data = '';
                res.on('data', (chunk) => {
                    data += chunk;
                });

                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        this.LoadModel(json);
                        console.log(`✅ Model loaded from URL: ${url} using direct download`);
                        resolve();
                    } catch (error) {
                        console.error('Failed to parse JSON:', error);
                        reject(error);
                    }
                });
            });

            req.on('error', (error) => {
                console.error('Request error:', error);
                reject(error);
            });

            req.end();
        } catch (error) {
            console.error('Error in direct download:', error);
            reject(error);
        }
    }

    // RL Methods
    getAction(state, possibleActions) {
        // Epsilon-greedy policy
        if (Math.random() < this.epsilon) {
            // Explore: choose random action
            return possibleActions[Math.floor(Math.random() * possibleActions.length)];
        } else {
            // Exploit: choose best action based on Q-values
            return this.getBestAction(state, possibleActions);
        }
    }

    getBestAction(state, possibleActions) {
        let bestAction = possibleActions[0];
        let bestValue = -Infinity;

        for (const action of possibleActions) {
            const stateAction = this.#getStateActionKey(state, action);
            let qValue;
            if (this.qTable.has(stateAction)) {
                qValue = this.qTable.get(stateAction);
            } else {
                // Use the neural network to estimate Q-value if not in Q-table
                qValue = this.run([...state, action])[0];
            }

            if (qValue > bestValue) {
                bestValue = qValue;
                bestAction = action;
            }
        }

        return bestAction;
    }

    remember(state, action, reward, nextState, done) {
        const encodedState = this.#encodeState(state);
        const encodedNextState = this.#encodeState(nextState);

        // Store experience in memory
        this.memory.push({ state: encodedState, action, reward, nextState: encodedNextState, done });

        // Limit memory size
        if (this.memory.length > this.maxMemory) {
            this.memory.shift();
        }
    }

    replay(batchSize = 32) {
        if (this.memory.length < batchSize) return;

        // Sample random experiences from memory
        const batch = this.#sampleMemory(batchSize);

        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            // Get current Q-value
            const stateAction = this.#getStateActionKey(state, action);
            const currentQ = this.qTable.has(stateAction) ? this.qTable.get(stateAction) : 0;

            // Calculate target Q-value
            let targetQ = reward;
            if (!done) {
                // Get max Q-value for next state
                const nextStateActions = this.#getPossibleActions(nextState);
                const nextBestAction = this.getBestAction(nextState, nextStateActions);
                const nextStateAction = this.#getStateActionKey(nextState, nextBestAction);
                const nextQ = this.qTable.has(nextStateAction) ? this.qTable.get(nextStateAction) : 0;

                targetQ += this.gamma * nextQ;
            }

            // Update Q-value
            this.qTable.set(stateAction, currentQ + this.learningRate * (targetQ - currentQ));
        }

        // Decay epsilon
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    saveModel(path) {
        const model = {
            network: this.toJSON(),
            qTable: Array.from(this.qTable.entries()),
            parameters: {
                gamma: this.gamma,
                epsilon: this.epsilon,
                epsilonMin: this.epsilonMin,
                epsilonDecay: this.epsilonDecay
            }
        };

        fs.writeFileSync(path, JSON.stringify(model));
    }

    loadModel(path) {
        const model = JSON.parse(fs.readFileSync(path, 'utf8'));

        this.fromJSON(model.network);
        this.qTable = new Map(model.qTable);

        // Load parameters
        this.gamma = model.parameters.gamma;
        this.epsilon = model.parameters.epsilon;
        this.epsilonMin = model.parameters.epsilonMin;
        this.epsilonDecay = model.parameters.epsilonDecay;
    }

    replay(batchSize = 32) {
        if (this.memory.length < batchSize) return;

        // Sample random experiences from memory
        const batch = this.#sampleMemory(batchSize);
        const states = [];
        const targets = [];

        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            const currentQ = this.run([...state, action])[0];
            let targetQ = reward;

            if (!done) {
                const nextActions = this.#getPossibleActions(nextState);
                const nextBestAction = this.getBestAction(nextState, nextActions);
                const nextQ = this.run([...nextState, nextBestAction])[0];
                targetQ += this.gamma * nextQ;
            }

            states.push([...state, action]);
            targets.push([targetQ]);
        }

        // Train the neural network
        this.train(states.map((s, i) => ({ input: s, output: targets[i] })), {
            iterations: 1,
            errorThresh: 0.01
        });

        // Decay epsilon
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    #getStateActionKey(state, action) {
        // Create a unique key for state-action pair
        return JSON.stringify({ state, action });
    }

    #sampleMemory(batchSize) {
        const samples = [];
        const memoryLength = this.memory.length;

        for (let i = 0; i < batchSize; i++) {
            const index = Math.floor(Math.random() * memoryLength);
            samples.push(this.memory[index]);
        }

        return samples;
    }

    #getPossibleActions(state) {
        // Implement a more sophisticated action space based on the state
        const baseActions = [0, 1, 2, 3]; // Basic action set
        const stateSpecificActions = this.#getStateSpecificActions(state);
        return [...new Set([...baseActions, ...stateSpecificActions])];
    }

    #getStateSpecificActions(state) {
        // Generate additional actions based on the current state
        const stateSpecificActions = [];

        // Example: If the state has a certain property, add a special action
        if (state.someProperty > 10) {
            stateSpecificActions.push(4); // Special action
        }

        // Example: Add actions based on state values
        for (let i = 0; i < state.length; i++) {
            if (state[i] > 0) {
                stateSpecificActions.push(5 + i); // Dynamic action based on state
            }
        }

        return stateSpecificActions;
    }

    #batchData(data, batchSize) {
        const batches = [];
        for (let i = 0; i < data.length; i += batchSize) {
            batches.push(data.slice(i, i + batchSize));
        }
        return batches;
    }

    #encodeState(state) {
        // Convert the state into a format suitable for the neural network
        // This method should be customized based on your specific state representation
        if (Array.isArray(state)) {
            return state.map(s => Number(s)); // Ensure all elements are numbers
        } else if (typeof state === 'object') {
            return Object.values(state).map(s => Number(s));
        } else {
            return [Number(state)];
        }
    }

    getAction(state, possibleActions) {
        const encodedState = this.#encodeState(state);

        // Epsilon-greedy policy
        if (Math.random() < this.epsilon) {
            // Explore: choose random action
            return possibleActions[Math.floor(Math.random() * possibleActions.length)];
        } else {
            // Exploit: choose best action based on Q-values
            return this.getBestAction(encodedState, possibleActions);
        }
    }



    #computeGradients(batch) {
        const gradients = math.zeros(this.net.weights.length);
        for (const example of batch) {
            const output = this.run(example.input);
            const error = math.subtract(example.output, output);
            const backpropGradient = this.#backpropagate(error);
            gradients = math.add(gradients, backpropGradient);
        }
        return math.divide(gradients, batch.length);
    }

    #backpropagate(error) {
        // تنفيذ الانتشار الخلفي باستخدام mathjs
        const layers = this.net.layers;
        let delta = error;
        const gradients = [];

        for (let i = layers.length - 1; i >= 0; i--) {
            const layer = layers[i];
            const activations = layer.weights ? math.multiply(layer.weights, delta) : delta;
            delta = math.dotMultiply(activations, this.#derivativeReLU(layer.weights));

            if (layer.weights) {
                const layerGradient = math.multiply(math.transpose(delta), layer.weights);
                gradients.unshift(layerGradient);
            }
        }

        return math.flatten(gradients);
    }

    #updateWeights(gradients) {
        if (this.optimizer === 'adam') {
            this.#adamOptimizer(gradients);
        } else {
            this.#sgdOptimizer(gradients);
        }
    }

    #adamOptimizer(gradients) {
        const beta1 = 0.9;
        const beta2 = 0.999;
        const epsilon = 1e-8;

        this.t += 1;

        this.m = math.add(math.multiply(beta1, this.m), math.multiply(1 - beta1, gradients));
        this.v = math.add(math.multiply(beta2, this.v), math.multiply(1 - beta2, math.dotMultiply(gradients, gradients)));

        const mHat = math.divide(this.m, 1 - Math.pow(beta1, this.t));
        const vHat = math.divide(this.v, 1 - Math.pow(beta2, this.t));

        const update = math.dotDivide(math.multiply(this.learningRate, mHat), math.add(math.sqrt(vHat), epsilon));

        this.net.weights = math.subtract(this.net.weights, update);
    }

    #sgdOptimizer(gradients) {
        this.net.weights = math.subtract(this.net.weights, math.multiply(this.learningRate, gradients));
    }

    #validateError(data) {
        const errors = data.map(example => {
            const output = this.run(example.input);
            return math.sum(math.abs(math.subtract(example.output, output)));
        });
        return math.mean(errors);
    }

    #argmax(array) {
        return array.indexOf(Math.max(...array));
    }

    #derivativeReLU(x) {
        return math.map(x, value => value > 0 ? 1 : 0);
    }
}

module.exports = DeepLearn;