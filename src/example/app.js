/**
 * Created by toanngo on 10/18/2015.
 */
$(function () {
    var raw_input = days_20[0].data; // this data has x as timestamp -> need to transform
    var processed_input = Regression.transformTimeStamp(raw_input, 0);

    var alpha = 0.001;
    var num_iters = 20000;

    var f = function (x) {
        return Math.pow(x, 0.5);
    };

    var r = Regression.init(processed_input);

    r.transform(f);

    var theta = r.process(alpha, num_iters);

    var predict = [];

    for (var i = 0; i < 40; i++) {
        predict.push(r.predict(i, theta));
    }

    $(function () {
        $('#container').highcharts({
                title: {
                    text: 'Predict Results'
                },
                //xAxis: {
                //    tickInterval: 1
                //},
                series: [
                    {
                        type: 'scatter',
                        lineWidth: 1,
                        name: 'Actual',
                        data: processed_input,
                        marker: {
                            radius: 4
                        }
                    },
                    {
                        type: 'scatter',
                        color: 'red',
                        lineWidth: 1,
                        name: 'Predictions',
                        data: predict,
                        marker: {
                            radius: 4,
                            fillColor: 'red'
                        }
                    }
                ]
            }
        );
    });
});