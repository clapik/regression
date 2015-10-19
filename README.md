# regression

A library to do regression on a set of data [x, y]

Usage:
```
// initialize the regression 
var r = Regression.init(processed_input);


// transform x if necessary by supplying a function f(x) -> new x
// e.g. instead of a straight line regression a + bx, you can have a + b sqrt(x) by supply f(x) -> sqrt(x)
r.transform(f);

// start processing the data
var theta = r.process();

// predict the next data point based on theta
var nextPoint = r.predict(nextX, theta);
```

screenshot:
![alt tag](https://raw.github.com/clapik/regression/master/src/example/sample.PNG)
