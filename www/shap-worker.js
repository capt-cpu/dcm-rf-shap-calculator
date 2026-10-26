"use strict";
// This dedicated local browser thread never sends patient values over the network.
self.window = self;
importScripts("../model/rf-model-data.js?v=20261025-mi20-v2", "rf-engine.js?v=20261025-mi20-v2");
const ready = self.RFEngine.create(self.RF_MODEL_DATA, {
  baseURL: new URL("../", self.location.href).href
}).then(engine => {
  const check = engine.verifyFixtures(1e-12);
  if (!check.passed) throw new Error("Worker model parity verification failed.");
  self.postMessage({ type: "ready", release: self.RF_MODEL_DATA.release,
    modelMd5: self.RF_MODEL_DATA.modelMd5 });
  return engine;
}).catch(error => {
  self.postMessage({ type: "error", message: error.message });
  throw error;
});
self.onmessage = async function (event) {
  const { id, values } = event.data;
  try {
    const engine = await ready;
    const result = await engine.exactShap(values, {
      yieldEvery: 0,
      onProgress: progress => self.postMessage({ type: "progress", id, progress })
    });
    self.postMessage({ type: "result", id, result });
  } catch (error) {
    self.postMessage({ type: "error", id, message: error.message });
  }
};
