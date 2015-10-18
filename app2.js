/**
 * Created by toanngo on 10/18/2015.
 */
$(function () {
    //var raw_input = data;
    var raw_input = data2[0].data; // this data has x as timestamp -> need to transform
    //var process_input = raw_input;
    var processed_input = Regression.transformTimeStamp(raw_input);

    var alpha = 0.001;
    var num_iters = 20000;

    var transform = function (x) {
        return Math.pow(x, 0.5);
    };

    //var theta = Regression.process(processed_input, alpha, num_iters, transform);

    var r = Regression.init(raw_input);

    console.log(r.n);
    console.log(r.x);
    console.log(r.y);
    console.log(r.theta);



    //var predict = [];
    //
    //for (var i = 0; i < 50; i++) {
    //    predict.push(Regression.getPoint(i, theta));
    //}
    //
    //$(function () {
    //    $('#container').highcharts({
    //        title: {
    //            text: 'Scatter plot with regression'
    //        },
    //        series: [
    //            {
    //                type: 'scatter',
    //                name: 'Observations',
    //                data: processed_input,
    //                marker: {
    //                    radius: 4
    //                }
    //            },
    //            {
    //                type: 'scatter',
    //                name: 'Observations',
    //                data: predict,
    //                marker: {
    //                    radius: 4,
    //                    fillColor: 'red'
    //                }
    //            }]
    //    });
    //});
});