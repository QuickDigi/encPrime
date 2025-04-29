// QunTime: Parallel & Entangled Mode Simulation in Node.js
// ---------------------------------------------------------
// Implements a parallel optimizer with "entanglement" via SharedArrayBuffer + Atomics.

const { Worker } = require('worker_threads');

class QunTime {
  static entangled = false;

  /**
   * Enable or disable Entangled Mode
   * @param {boolean} enable
   */
  static entangledMode(enable = true) {
    this.entangled = enable;
  }

  /**
   * Executes a function in parallel across multiple workers.
   *
   * @param {Function} fn - Function to execute. Receives (sharedBest, maybeUpdateShared) if Entangled Mode is enabled.
   * @param {number} [count=4] - Number of parallel workers to spawn.
   * @returns {Promise<number[]>} Resolves with an array of worker results.
   *
   * @example
   * // Enable Entangled Mode (shared optimization)
   * QunTime.entangledMode(true);
   *
   * (async () => {
   *   const startTime = Date.now();
   *   
   *   // Define a simple function to run
   *   const sumFn = () => 15 + 6;
   *
   *   // Execute the function in 32 workers
   *   const results = await QunTime.parallel(sumFn, 32);
   *   
   *   const endTime = Date.now();
   *   const bestResult = Math.min(...results);
   *
   *   console.log('Quantum Solve Time:', (endTime - startTime) + 'ms');
   *   console.log('Best Approximate Solution:', bestResult);
   * })();
   */
  static async parallel(fn, count = 4) {
    return new Promise((resolve, reject) => {
      let completed = 0;
      const results = [];

      const fnString = fn.toString();
      const entangledEnabled = this.entangled;

      // Setup shared buffer if entangled
      let sharedBuffer, sharedArray;
      if (entangledEnabled) {
        sharedBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
        sharedArray = new Int32Array(sharedBuffer);
        sharedArray[0] = Number.MAX_SAFE_INTEGER; // Initialize to max
      }

      for (let i = 0; i < count; i++) {
        const worker = new Worker(
          `
          const { parentPort, workerData } = require('worker_threads');
          const fn = ${fnString};

          // Setup shared array if entangled
          const entangled = workerData.entangled;
          let sharedArray = null;
          if (entangled && workerData.sharedBuffer) {
            sharedArray = new Int32Array(workerData.sharedBuffer);
          }

          function maybeUpdateShared(score) {
            if (!sharedArray) return;
            const scaled = Math.floor(score * 1e6);
            const current = Atomics.load(sharedArray, 0);
            if (scaled < current) {
              Atomics.store(sharedArray, 0, scaled);
            }
          }

          // Read starting best from shared
          const startingBest = sharedArray ? (Atomics.load(sharedArray, 0) / 1e6) : Infinity;

          const result = fn(startingBest, maybeUpdateShared);
          parentPort.postMessage(result);
        `,
          {
            eval: true,
            workerData: {
              entangled: entangledEnabled,
              sharedBuffer: entangledEnabled ? sharedBuffer : null
            }
          }
        );

        worker.on('message', result => {
          results.push(result);
          completed++;
          if (completed === count) resolve(results);
        });

        worker.on('error', reject);
      }
    });
  }
}

module.exports = QunTime;
