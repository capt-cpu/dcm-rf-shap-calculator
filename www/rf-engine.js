(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RFEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function decodeForest(bytes, featureCount) {
    const data = bytes instanceof ArrayBuffer
      ? new DataView(bytes)
      : new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 0;
    const integer = () => { const value = data.getInt32(offset, true); offset += 4; return value; };
    const number = () => { const value = data.getFloat64(offset, true); offset += 8; return value; };
    if (integer() !== 0x32464D52 || integer() !== 1 || integer() !== featureCount) {
      throw new Error("The RF asset format does not match this calculator.");
    }
    const treeCount = integer();
    const positiveClass = integer();
    if (treeCount !== 300 || positiveClass !== 1) throw new Error("Unexpected RF class mapping or tree count.");
    const means = Array.from({ length: featureCount }, number);
    const standardDeviations = Array.from({ length: featureCount }, number);
    if (!means.every(Number.isFinite) || !standardDeviations.every(v => Number.isFinite(v) && v > 0)) {
      throw new Error("Invalid frozen preprocessing parameters.");
    }
    const trees = [];
    for (let i = 0; i < treeCount; i += 1) {
      const n = integer();
      if (n < 1 || n > 10000) throw new Error("Invalid tree length.");
      const tree = {};
      for (const name of ["l", "r", "v"]) tree[name] = Int32Array.from({ length: n }, integer);
      tree.s = Float64Array.from({ length: n }, number);
      tree.p = Int32Array.from({ length: n }, integer);
      for (let node = 0; node < n; node += 1) {
        if (!Number.isFinite(tree.s[node]) || tree.v[node] < 0 || tree.v[node] > featureCount) {
          throw new Error("Invalid tree split.");
        }
        if (tree.v[node] === 0) {
          if (![1, 2].includes(tree.p[node])) throw new Error("Invalid terminal class.");
        } else if (tree.l[node] <= node + 1 || tree.r[node] <= node + 1 || tree.l[node] > n || tree.r[node] > n) {
          throw new Error("Invalid tree child index.");
        }
      }
      trees.push(tree);
    }
    if (offset !== data.byteLength) throw new Error("RF asset has trailing or missing data.");
    return { means, standardDeviations, positiveClass, trees };
  }

  async function create(model, options) {
    if (!model || model.schemaVersion !== 2 || !Array.isArray(model.features)
      || model.modelCount !== 20 || model.positiveClass !== "Bad"
      || !Array.isArray(model.forestFiles) || model.forestFiles.length !== 20) {
      throw new Error("MI20 RF model data are incomplete.");
    }
    const config = options || {};
    const loadForest = config.loadForest || (async function (path) {
      if (typeof DecompressionStream !== "function") {
        throw new Error("This calculator requires a current version of Chrome, Edge, Firefox or Safari.");
      }
      const assetURL = config.baseURL ? new URL(path, config.baseURL).href : path;
      const response = await fetch(assetURL, { cache: "force-cache" });
      if (!response.ok) throw new Error("A frozen RF asset could not be loaded. Please refresh and retry.");
      const compressed = await response.arrayBuffer();
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).arrayBuffer();
    });

    const featureCount = model.features.length;
    const fullMask = (1 << featureCount) - 1;
    const factorial = [1];
    for (let i = 1; i <= featureCount; i += 1) factorial[i] = factorial[i - 1] * i;

    const forests = await Promise.all(model.forestFiles.map(async (path) =>
      decodeForest(await loadForest(path), featureCount)));
    if (forests.reduce((n, forest) => n + forest.trees.length, 0) !== model.treeCount) {
      throw new Error("The complete 20-fit RF ensemble was not loaded.");
    }

    function standardize(raw, forest) {
      if (!Array.isArray(raw) || raw.length !== featureCount) {
        throw new Error("The number of patient predictors does not match the locked model.")
      }
      if (!raw.every(value => typeof value === "number" && Number.isFinite(value))) {
        throw new Error("All seven complete numeric inputs are required; missing values are not imputed.");
      }
      return raw.map((value, index) => (value - forest.means[index]) / forest.standardDeviations[index]);
    }

    function predictStandardized(values, forest) {
      let badVotes = 0;
      for (let treeIndex = 0; treeIndex < forest.trees.length; treeIndex += 1) {
        const tree = forest.trees[treeIndex];
        let node = 0;
        while (tree.v[node] !== 0) {
          const featureIndex = tree.v[node] - 1;
          node = values[featureIndex] <= tree.s[node]
            ? tree.l[node] - 1
            : tree.r[node] - 1;
        }
        if (tree.p[node] === forest.positiveClass) badVotes += 1;
      }
      return badVotes / forest.trees.length;
    }

    function predict(raw) {
      return forests.reduce((sum, forest) =>
        sum + predictStandardized(standardize(raw, forest), forest), 0) / forests.length;
    }

    const standardizedBackground = forests.map(forest => model.background.map(row => standardize(row, forest)));
    const baselineRisk = model.background.reduce((sum, row) => sum + predict(row), 0) / model.background.length;

    function verifyFixtures(tolerance) {
      const allowed = tolerance == null ? 1e-12 : tolerance;
      let maximumError = 0;
      for (const fixture of model.fixtures) {
        maximumError = Math.max(maximumError, Math.abs(predict(fixture.x) - fixture.p));
      }
      const baseline = baselineRisk;
      maximumError = Math.max(
        maximumError,
        Math.abs(baseline - model.backgroundExpectedRisk)
      );
      return {
        passed: maximumError <= allowed,
        maximumError,
        fixtures: model.fixtures.length,
        baseline
      };
    }

    async function exactShap(raw, options) {
      const config = options || {};
      const onProgress = typeof config.onProgress === "function"
        ? config.onProgress
        : function () {};
      const yieldEvery = Number.isFinite(config.yieldEvery) ? config.yieldEvery : 8;
      const yieldFunction = typeof config.yieldFunction === "function"
        ? config.yieldFunction
        : function () { return Promise.resolve(); };

      const patients = forests.map(forest => standardize(raw, forest));
      const coalitionValues = new Float64Array(1 << featureCount);

      for (let mask = 0; mask <= fullMask; mask += 1) {
        if (mask === 0) {
          coalitionValues[mask] = baselineRisk;
          continue;
        }
        if (mask === fullMask) {
          coalitionValues[mask] = predict(raw);
          continue;
        }
        let total = 0;
        for (let fitIndex = 0; fitIndex < forests.length; fitIndex += 1) {
          const forest = forests[fitIndex];
          for (let backgroundIndex = 0; backgroundIndex < model.background.length; backgroundIndex += 1) {
            const hybrid = standardizedBackground[fitIndex][backgroundIndex].slice();
            for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
              if (mask & (1 << featureIndex)) hybrid[featureIndex] = patients[fitIndex][featureIndex];
            }
            total += predictStandardized(hybrid, forest);
          }
        }
        coalitionValues[mask] = total / (forests.length * model.background.length);

        if (yieldEvery > 0 && mask % yieldEvery === 0) {
          onProgress((mask + 1) / (fullMask + 1));
          await yieldFunction();
        }
      }

      // Reproduce kernelshap 0.7.0 exact mode. It fits the complete weighted
      // coalition system under the efficiency constraint, using the same
      // normalized Kernel SHAP weights and compound-symmetric A matrix.
      function choose(n, k) {
        return factorial[n] / (factorial[k] * factorial[n - k]);
      }
      const rawSizeWeights = new Float64Array(featureCount + 1);
      let rawWeightSum = 0;
      for (let subsetSize = 1; subsetSize < featureCount; subsetSize += 1) {
        rawSizeWeights[subsetSize] = (featureCount - 1)
          / (choose(featureCount, subsetSize) * subsetSize * (featureCount - subsetSize));
        rawWeightSum += rawSizeWeights[subsetSize];
      }
      const sizeWeights = new Float64Array(featureCount + 1);
      const coalitionWeights = new Float64Array(featureCount + 1);
      for (let subsetSize = 1; subsetSize < featureCount; subsetSize += 1) {
        sizeWeights[subsetSize] = rawSizeWeights[subsetSize] / rawWeightSum;
        coalitionWeights[subsetSize] = sizeWeights[subsetSize]
          / choose(featureCount, subsetSize);
      }

      const bVector = new Float64Array(featureCount);
      for (let mask = 1; mask < fullMask; mask += 1) {
        let subsetSize = 0;
        for (let bit = 0; bit < featureCount; bit += 1) {
          if (mask & (1 << bit)) subsetSize += 1;
        }
        const weightedDelta = coalitionWeights[subsetSize]
          * (coalitionValues[mask] - coalitionValues[0]);
        for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
          if (mask & (1 << featureIndex)) bVector[featureIndex] += weightedDelta;
        }
      }

      let offDiagonal = 0;
      for (let subsetSize = 1; subsetSize < featureCount; subsetSize += 1) {
        offDiagonal += sizeWeights[subsetSize]
          * subsetSize * (subsetSize - 1)
          / featureCount / (featureCount - 1);
      }
      const identityCoefficient = 0.5 - offDiagonal;
      const commonCoefficient = offDiagonal;
      const inverseIdentity = 1 / identityCoefficient;
      const inverseCommon = -commonCoefficient /
        (identityCoefficient * (identityCoefficient + featureCount * commonCoefficient));

      function applyAInverse(vector) {
        const vectorSum = Array.from(vector).reduce((sum, value) => sum + value, 0);
        return Array.from(vector, (value) =>
          inverseIdentity * value + inverseCommon * vectorSum
        );
      }

      const inverseB = applyAInverse(bVector);
      const constraint = coalitionValues[fullMask] - coalitionValues[0];
      const sumInverseA = featureCount /
        (identityCoefficient + featureCount * commonCoefficient);
      const lagrangeAdjustment = (
        inverseB.reduce((sum, value) => sum + value, 0) - constraint
      ) / sumInverseA;
      const adjustedB = Array.from(bVector, (value) => value - lagrangeAdjustment);
      const shapValues = applyAInverse(adjustedB);

      onProgress(1);
      const baseline = coalitionValues[0];
      const probability = coalitionValues[fullMask];
      const shapSum = shapValues.reduce((sum, value) => sum + value, 0);

      return {
        baseline,
        probability,
        shapValues,
        shapSum,
        reconstructedProbability: baseline + shapSum,
        additivityError: probability - baseline - shapSum
      };
    }

    return {
      model,
      standardize,
      predict,
      verifyFixtures,
      exactShap
    };
  }

  return { create };
});
