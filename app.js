$(function () {
    // x is a vector(n x 1)
    // theta is a vector(n x 1);
    var hypothesis = function (x, theta) {
        return theta.dot(x)
    };

    var computeCost = function (x, y, theta) { // vector x and vector y
        var m = x.rows();
        var sum = 0;
        for (var i = 1; i <= m; i++) {
            sum += Math.pow(hypothesis(x.row(i), theta) - y.e(i), 2);
        }
        return sum / (2 * m);
    };

    var addOnesToLeft = function (x) {
        var ones = [];
        for (var i = 0; i < x.dimensions(); i++) {
            ones.push([1]);
        }
        return $M(ones).augment(x);
    };

    // x is a matrix
    // y is a vector(n x 1)
    // theta is a vector(n x 1), but created as matrix (n x 1) to utilize transpose
    var gradientDescent = function (x, y, theta, alpha, num_iterations) {
        var computedCost = [];

        var m = x.rows();
        for (var iteration = 0; iteration < num_iterations; iteration++) {
            for (var j = 1; j <= theta.dimensions(); j++) {
                var innerSum = 0;
                for (var i = 1; i <= m; i++) {
                    innerSum += (hypothesis(x.row(i), theta) - y.e(i)) * x.e(i, j);
                }
                var newVal = theta.e(j) - (alpha / m) * innerSum;
                theta.setE(j, newVal);
                computedCost.push(computeCost(x, y, theta));
            }
        }
        console.log(computedCost)
        return theta;
    };

    // normalize a feature x, which is vector
    var normalizeInput = function (x, num_features) {
        //var array = x.getArray();
        var array = x;
        for(var num_fea = 0; num_fea < num_features; num_fea++) {
            var mean, min, max, sum;
            sum = 0;
            min = Number.POSITIVE_INFINITY;
            max = Number.NEGATIVE_INFINITY;
            for (var i = 0; i < array.length; i++) {
                sum += array[i][num_fea];
                if (array[i][num_fea] > max) max = array[i][num_fea];
                if (array[i][num_fea] < min) min = array[i][num_fea];
            }

            mean = sum / array.length;

            for (i = 0; i < array.length; i++) {
                array[i][num_fea] = (array[i][num_fea] - mean) / (max - min)
            }
        }
        return array;
    };

    var normalizeDateTime = function(x) {
        var array = x;
        var mean, min, max, sum;
        sum = 0;
        min = Number.POSITIVE_INFINITY;
        max = Number.NEGATIVE_INFINITY;
        for (var i = 0; i < array.length; i++) {
            sum += array[i][0];
            if (array[i][0] > max) max = array[i][0];
            if (array[i][0] < min) min = array[i][0];
        }

        mean = sum / array.length;

        for (i = 0; i < array.length; i++) {
            //array[i][0] = (array[i][0] - mean) / (max - min)
            array[i][0] = (array[i][0] - min) / (24 * 3600 * 1000)
        }

        console.log(array);

        return array;
    };

    var addFeatureRootX = function(data) {
        var newData = $.extend(true, [], data);
        // data will have (x,y), we want (x, root(x), y)
        for(var i = 0; i < newData.length; i++) {
            newData[i].splice(1, 0, Math.sqrt(newData[i][0]))
        }
        return newData;
    };

    // test cases
    var original = data2[0].data;


    var modified = addFeatureRootX(original);

    console.log(original);
    console.log(modified);

    modified = normalizeInput(modified, 2); // 2 features
    original = normalizeInput(original, 1);

    var input = $M(modified);

    var x = addOnesToLeft(input.col(1));
    var y = input.col(3);
    var theta = $V([0, 1]);


    var result = gradientDescent(x, y, theta, 0.001, 2000);

    var theta0 = result.e(1);
    var theta1 = result.e(2);

    var getNextPoint = function (next, theta0, theta1) {
        return [next, theta0 + next * theta1];
    };

    console.log("RESULT")
    console.log(result)

    $(function () {
        $('#container').highcharts({
            title: {
                text: 'Scatter plot with regression line'
            },
            series: [
                {
                    type: 'line',
                    name: 'Regression Line',
                    data: [getNextPoint(-0.5, theta0, theta1), getNextPoint(0.5, theta0, theta1)],
                    marker: {
                        enabled: false
                    },
                    states: {
                        hover: {
                            lineWidth: 0
                        }
                    },
                    enableMouseTracking: false
                },
                {
                    type: 'scatter',
                    name: 'Observations',
                    data: original,
                    marker: {
                        radius: 4
                    }
                }]
        });
    });
});