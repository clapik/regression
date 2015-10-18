function Regression() {
}

Regression.nData = [];

Regression.transformFunc = false;

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

// raw_input example: [[0,0], [1,1]] -> x1, x2,...., xn, y
Regression.prototype.process = function (alpha, num_iters, transform) {
    var input_matrix = $M(raw_input);

    // num features
    var n = input_matrix.dimensions().cols - 1;

    // Get parameters
    // feature matrix x
    var x = this.getX(input_matrix, n);

    // do we transform the matrix?
    if (transform) {
        this.transformFunc = transform;
        x = this.transformX(x, this.transformFunc);
    }

    //x = addRoot2Feature(x);
    //
    //n += 1; // because we added a feature

    // normalize features to (1,1)
    this.nData = this.getNormalizationData(x);
    x = this.normalizeFeatures(x);

    var y = this.getY(input_matrix);
    var theta = Vector.Zero(n + 1); // remember there's a ones in front

    return this.gradientDescent(x, y, theta, alpha, num_iters);
};

Regression.transformX = function (x, transform) {
    return this.getOnes(x.dimensions().rows).augment(x.col(2).map(function (value) {
        return transform(value);
    }));
};

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

// x is a vector! not a matrix in this case
Regression.hypothesis = function (x, theta) {
    return x.dot(theta)
};

Regression.computeCost = function (x, y, theta) {
    var m = y.dimensions(); // num rows

    var innerSum = 0;
    for (var i = 1; i <= m; i++) {
        innerSum += Math.pow((this.hypothesis(x.row(i), theta) - y.e(i)), 2);
    }

    return innerSum / (2 * m);
};

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
            theta.setE(j, newThetaValue);
            computedCosts.push(this.computeCost(x, y, theta));
        }

        num_iters--;
    }

    return theta;
};

Regression.getPoint = function (x, theta) {
    if (!this.transformFunc) {
        return [x, this.hypothesis($V([1, this.normalizeX(x)]), theta)];
    }
    else {
        return [x, this.hypothesis($V([1, this.normalizeX(this.transformFunc(x))]), theta)];
    }
};

// input as an array with x = timestamp. Only works with [x,y] pair for now.
Regression.transformTimeStamp = function (input) {
    var copy = this.deepArrayCopy(input);

    var min = Number.POSITIVE_INFINITY;
    var max = Number.NEGATIVE_INFINITY;

    for (var i = 0; i < copy.length; i++) {
        if (copy[i][0] > max) max = copy[i][0];
        if (copy[i][0] < min) min = copy[i][0];
    }

    for (i = 0; i < copy.length; i++) {
        copy[i][0] = (copy[i][0] - min) / (24 * 3600 * 1000)
    }

    return copy;
};

Regression.deepArrayCopy = function (array) {
    return $.extend(true, [], array);
};

Regression.normalizeFeatures = function (x) {
    var n = x.dimensions().cols; // num features
    var rows = x.dimensions().rows; // num rows

    var newX = this.getOnes(rows);

    for (var i = 2; i <= n; i++) { // normalize each feature; remember x starts with 1 -> ignore 1
        var min = this.nData[i - 2][0];
        var max = this.nData[i - 2][1];
        var mean = this.nData[i - 2][2];

        var copy = this.deepArrayCopy(x.col(i).getArray()); // get the array from vector x[i]

        // start normalizing
        for (var j = 0; j < copy.length; j++) {
            copy[j] = (copy[j] - mean) / (max - min);
        }

        newX = newX.augment($M(copy));
    }

    return newX;
};

// used for de-normalize a single value of x (not array/vector/matrix)
Regression.denormalizeX = function (x) {
    var min = this.nData[0][0];
    var max = this.nData[0][1];
    var mean = this.nData[0][2];
    return x * (max - min) + mean;
};

Regression.normalizeX = function (x) {
    var min = this.nData[0][0];
    var max = this.nData[0][1];
    var mean = this.nData[0][2];
    return (x - mean) / (max - min);
};

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
        result.push([min, max, mean]);
    }

    return result;
};