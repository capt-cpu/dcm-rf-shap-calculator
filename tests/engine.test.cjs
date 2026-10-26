"use strict";
// Public regression tests contain synthetic fixtures only, never patient records.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const zlib = require("node:zlib");
const assert = require("node:assert/strict");
const root = path.resolve(__dirname, "..");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, "model/rf-model-data.js"), "utf8"), context);
const model = JSON.parse(JSON.stringify(context.window.RF_MODEL_DATA));
const { create } = require(path.join(root, "www/rf-engine.js"));
(async () => {
  const engine = await create(model, {
    loadForest: async file => zlib.gunzipSync(fs.readFileSync(path.join(root, file)))
  });
  const verification = engine.verifyFixtures(1e-12);
  assert(verification.passed);
  assert.equal(model.modelCount, 20);
  assert.equal(model.treeCount, 6000);
  assert.equal(model.threshold, .5);
  for (const fixture of model.shapVerification) {
    const result = await engine.exactShap(fixture.x, { yieldEvery: 0 });
    assert(Math.abs(result.probability - fixture.p) < 1e-12);
    assert(Math.abs(result.baseline - fixture.baseline) < 1e-12);
    result.shapValues.forEach((v, i) => assert(Math.abs(v - fixture.values[i]) < 1e-9));
    assert(Math.abs(result.additivityError) < 1e-10);
  }
  for (const x of [[], [12, 0, 0, 0, 13, .3, null], [12, 0, 0, 0, 13, .3, NaN]]) {
    assert.throws(() => engine.predict(x));
  }
  console.log(JSON.stringify({ release: model.release, ...verification, exactShapPassed: true }));
})().catch(error => { console.error(error); process.exitCode = 1; });
