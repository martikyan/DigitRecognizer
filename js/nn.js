let brain = require('brain.js');
let fs    = require('fs');

var dataFileBuffer  = fs.readFileSync(__dirname + '/../training_data/t10k-images.idx3-ubyte');
var labelFileBuffer = fs.readFileSync(__dirname + '/../training_data/t10k-labels.idx1-ubyte');

var pixelValues = getPixels(dataFileBuffer, labelFileBuffer);

var net = new brain.NeuralNetwork({
  activation: 'sigmoid', // activation function
  hiddenLayers: [10]
});

net.train(pixelValues, {
  errorThresh: 0.001,  // error threshold to reach
  iterations: 500,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 100,       // number of iterations between logging
  learningRate: 0.3    // learning rate
});

for(var i = 0; i < 10; i++){
	console.log('expected is ' + JSON.stringify(maxValueJSON(pixelValues[i].output)));
	var output = net.run(pixelValues[i].input);
	console.log('actual   is ' + JSON.stringify(maxValueJSON(output)) + '\n');
}
function getPixels(dataFileBuffer, labelFileBuffer){
	var _pixelValues = [];
	// It would be nice with a checker instead of a hard coded 200 limit here
	for (var image = 0; image <= 200; image++) {
		var digits = {
			"0": 0,
			"1": 0,
			"2": 0,
			"3": 0,
			"4": 0,
			"5": 0,
			"6": 0,
			"7": 0,
			"8": 0,
			"9": 0
		};
	    var pixels = [];

	    for (var x = 0; x <= 27; x++) {
	        for (var y = 0; y <= 27; y++) {
	            pixels.push((dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]) / 255);
	        }
	    }
	    digits[JSON.stringify(labelFileBuffer[image + 8])] = 1;
	    var imageData  = {input : pixels, output: digits};
	    //imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

	    _pixelValues.push(imageData);
	}
	return _pixelValues;
}

function maxValueJSON(_json){	
	var tempKey = 0,
	tempValue = 0,
	result = {};

	for (key in _json)
		if(_json[key] > tempValue){
			tempKey = key;
			tempValue = _json[key];
		}
		
	result[tempKey] = tempValue;
	return result;
}