(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RFEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function create(model) {
    if (!model || !Array.isArray(model.features) || !Array.isArray(model.trees)) {
      throw new Error("RF模型数据不完整。")
    }

    const featureCount = model.features.length;
    const fullMask = (1 << featureCount) - 1;
    const factorial = [1];
    for (let i = 1; i <= featureCount; i += 1) factorial[i] = factorial[i - 1] * i;

    function standardize(raw) {
      if (!Array.isArray(raw) || raw.length !== featureCount) {
        throw new Error("患者指标数量与锁定模型不一致。")
      }
      return raw.map((value, index) =>
        (Number(value) - model.means[index]) / model.standardDeviations[index]
      );
    }

    function predictStandardized(values) {
      let yesVotes = 0;
      for (let treeIndex = 0; treeIndex < model.trees.length; treeIndex += 1) {
        const tree = model.trees[treeIndex];
        let node = 0;
        while (tree.v[node] !== 0) {
          const featureIndex = tree.v[node] - 1;
          node = values[featureIndex] <= tree.s[node]
            ? tree.l[node] - 1
            : tree.r[node] - 1;
        }
        if (tree.p[node] === 2) yesVotes += 1;
      }
      return yesVotes / model.trees.length;
    }

    function predict(raw) {
      return predictStandardized(standardize(raw));
    }

    const standardizedBackground = model.background.map(standardize);

    function verifyFixtures(tolerance) {
      const allowed = tolerance == null ? 1e-12 : tolerance;
      let maximumError = 0;
      for (const fixture of model.fixtures) {
        maximumError = Math.max(maximumError, Math.abs(predict(fixture.x) - fixture.p));
      }
      const baseline = standardizedBackground.reduce(
        (sum, row) => sum + predictStandardized(row), 0
      ) / standardizedBackground.length;
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

      const patient = standardize(raw);
      const coalitionValues = new Float64Array(1 << featureCount);

      for (let mask = 0; mask <= fullMask; mask += 1) {
        let total = 0;
        for (let backgroundIndex = 0; backgroundIndex < standardizedBackground.length; backgroundIndex += 1) {
          const hybrid = standardizedBackground[backgroundIndex].slice();
          for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
            if (mask & (1 << featureIndex)) hybrid[featureIndex] = patient[featureIndex];
          }
          total += predictStandardized(hybrid);
        }
        coalitionValues[mask] = total / standardizedBackground.length;

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
      predictStandardized,
      verifyFixtures,
      exactShap
    };
  }

  return { create };
});
