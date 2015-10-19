function Regression() {
}

// constructor
Regression.init = function (raw_input) {
    var R = new Regression();

    var input_matrix = $M(raw_input);

    // num features
    R.n = input_matrix.dimensions().cols - 1;

    // Get parameters
    // feature matrix x, with 1s in front
    R.x = this.getX(input_matrix, R.n);
    // get result vector
    R.y = this.getY(input_matrix);
    // init theta to all 0s
    R.theta = Vector.Zero(R.n + 1); // remember there's a ones in front
    return R;
};

/**
 * INSTANCE METHODS
 */

Regression.prototype.transform = function (f) {
    this.f = f;
    this.x = Regression.getOnes(this.x.dimensions().rows).augment(this.x.col(2).map(function (value) {
        return f(value);
    }));
};

// Normalize the feature matrix x
Regression.prototype.normalize = function () {
    // get normalization data, which will be used later to predict results
    this.nData = Regression.getNormalizationData(this.x);
    this.x = Regression.normalizeFeatures(this.x, this.nData);
};

Regression.prototype.process = function (alpha, num_iters) {
    // normalize otherwise it's gonna be hard to get result
    this.normalize();
    return Regression.gradientDescent(this.x, this.y, this.theta, alpha, num_iters);
};

Regression.prototype.predict = function (x, theta) {
    if (!this.f) {
        return [x, Regression.hypothesis($V([1, this.normalizeX(x)]), theta)];
    }
    else {
        return [x, Regression.hypothesis($V([1, this.normalizeX(this.f(x))]), theta)];
    }
};

// used for de-normalize a single value of x (not array/vector/matrix)
Regression.prototype.denormalizeX = function (x) {
    var min = this.nData[0].min;
    var max = this.nData[0].max;
    var mean = this.nData[0].mean;
    return x * (max - min) + mean;
};

Regression.prototype.normalizeX = function (x) {
    var min = this.nData[0].min;
    var max = this.nData[0].max;
    var mean = this.nData[0].mean;
    return (x - mean) / (max - min);
};

/**
 * CLASS METHODS
 *
 * These methods must be immutable
 */

// need to add a column of 1's in front of x matrix
Regression.getX = function (input_matrix, num_features) {
    var num_rows = input_matrix.dimensions().rows;

    var x = this.getOnes(num_rows);

    for (var i = 1; i <= num_features; i++) {
        x = x.augment(input_matrix.col(i));
    }

    return x;
};

Regression.getY = function (input_matrix) {
    return input_matrix.col(input_matrix.dimensions().cols);
};

Regression.getOnes = function (num_rows) {
    var ones = [];
    for (var i = 0; i < num_rows; i++) {
        ones.push([1]);
    }
    return $M(ones);
};

Regression.addRoot2Feature = function (x) { // x contains [1,xi]...
    return x.augment(x.col(2).map(function (e) {
        return Math.sqrt(e);
    }))
};

// x is a vector! not a matrix in this case. Take the scalar product and we're good
Regression.hypothesis = function (x, theta) {
    return x.dot(theta)
};

// compute the cost of a hypothesis, given a specific theta
Regression.computeCost = function (x, y, theta) {
    var m = y.dimensions(); // num rows

    var innerSum = 0;
    for (var i = 1; i <= m; i++) {
        innerSum += Math.pow((this.hypothesis(x.row(i), theta) - y.e(i)), 2);
    }

    return innerSum / (2 * m);
};

// gradually reduce the cost function. Note: x should be normalized,
// alpha should be relatively small to x, and num_iters should be in thousands.
Regression.gradientDescent = function (x, y, theta, alpha, num_iters) {
    var m = y.dimensions(); // num rows

    var computedCosts = []; // keep track of the computed costs so far

    while (num_iters > 0) {
        for (var j = 1; j <= theta.dimensions(); j++) { // loop through theta values
            var innerSum = 0;

            for (var i = 1; i <= m; i++) {
                innerSum += (this.hypothesis(x.row(i), theta) - y.e(i)) * x.e(i, j);
            }

            var newThetaValue = theta.e(j) - innerSum * alpha / m;

            theta = theta.setE(j, newThetaValue);

            // keep track of the computed costs. Very useful for 2 reasons: 1) pick another alpha if computed costs
            // do not decrease as iteration goes, and 2) to compare different theta for different feature set
            computedCosts.push(this.computeCost(x, y, theta));
        }

        num_iters--;
    }

    return theta;
};

// input as an array with x = timestamp. Only works with [x,y] pair for now.
// location denotes where in the array the date time is
// this method will normalize date time into 1, 2, 3, 4, 5...
Regression.transformTimeStamp = function (input, location) {
    var copy = this.deepArrayCopy(input);

    var min = Number.POSITIVE_INFINITY;
    var max = Number.NEGATIVE_INFINITY;

    for (var i = 0; i < copy.length; i++) {
        if (copy[i][location] > max) max = copy[i][location];
        if (copy[i][location] < min) min = copy[i][location];
    }

    for (i = 0; i < copy.length; i++) {
        copy[i][location] = (copy[i][location] - min) / (24 * 3600 * 1000)
    }

    return copy;
};

// immutable method for array copy
Regression.deepArrayCopy = function (array) {
    return $.extend(true, [], array);
};

// normalize ALL features in feature matrix x, with a specific normalization data
// normalization is EXTREMELY IMPORTANT, especially when it involves exponents, root, invert, etc
Regression.normalizeFeatures = function (x, nData) {
    var n = x.dimensions().cols; // num features
    var rows = x.dimensions().rows; // num rows

    var newX = this.getOnes(rows);

    // normalize each feature; remember x starts with 1 -> ignore 1
    for (var i = 2; i <= n; i++) {
        var min = nData[i - 2].min;
        var max = nData[i - 2].max;
        var mean = nData[i - 2].mean;

        // get the array from vector x[i]
        var copy = this.deepArrayCopy(x.col(i).getArray());

        // start normalizing
        for (var j = 0; j < copy.length; j++) {
            copy[j] = (copy[j] - mean) / (max - min);
        }

        newX = newX.augment($M(copy));
    }

    return newX;
};

// for a specific feature matrix x, i want to retrieve its normalization data
// the result should be nData = [{minX1, maxX1, meanX1}, {minX2, maxX2, meanX2}, ...]
Regression.getNormalizationData = function (x) {
    var n = x.dimensions().cols; // num features
    var rows = x.dimensions().rows; // num rows

    var result = []; // each element will contain [min, max, mean]

    for (var i = 2; i <= n; i++) { // normalize each feature; remember x starts with 1 -> ignore 1
        var copy = this.deepArrayCopy(x.col(i).getArray()); // get the array from vector x[i]

        var min = Number.POSITIVE_INFINITY;
        var max = Number.NEGATIVE_INFINITY;
        var sum = 0;
        var mean;

        for (var j = 0; j < copy.length; j++) {
            sum += copy[j];
            if (copy[j] > max) max = copy[j];
            if (copy[j] < min) min = copy[j];
        }

        mean = sum / rows;
        result.push({min: min, max: max, mean: mean});
    }

    return result;
};