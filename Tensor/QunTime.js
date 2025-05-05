// QunTime: Parallel & Entangled Mode Simulation in Node.js
// ---------------------------------------------------------
// Implements a parallel optimizer with "entanglement" via SharedArrayBuffer + Atomics.

const { Worker } = require('worker_threads');

class QunTime {
  static entangled = false;

  /**
   * Enables or disables the Entangled Mode for parallel optimization.
   * When enabled, this mode allows workers to share and update a common best solution.
   * 
   * @param {boolean} [enable=true] - Whether to enable (true) or disable (false) Entangled Mode.
   * @returns {void}
   */
  static entangledMode(enable = true) {
    this.entangled = enable;
  }

  /**
   * Executes a function in parallel across multiple worker threads.
   * If entangled mode is enabled, workers can share and update a common best solution.
   *
   * @param {Function} fn - The function to be executed in parallel. It should accept two parameters:
   *                        1. startingBest: The initial best score (Infinity if not in entangled mode)
   *                        2. maybeUpdateShared: A function to update the shared best score (no-op if not in entangled mode)
   * @param {number} [count=4] - The number of worker threads to spawn. Defaults to 4.
   * @returns {Promise<Array>} A promise that resolves to an array of results from all worker threads.
   * @throws {Error} If any worker thread encounters an error during execution.
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
