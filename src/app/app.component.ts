import { Component, OnInit, ViewChild } from '@angular/core';
import { DrawableDirective } from './drawable.directive';

import * as tf from '@tensorflow/tfjs'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  title = 'tensorApp';
  linearModel: tf.Sequential;
  prediction : any;

  model: tf.Model;
  predictions: any;

  @ViewChild(DrawableDirective) canvas;

  ngOnInit () {
    this.trainNewModel()
    this.loadModel()
  }

  async trainNewModel () {
    this.linearModel = tf.sequential()

    this.linearModel.add(tf.layers.dense({
      units: 1,
      inputShape: [1]
    }))

    this.linearModel.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    })

    const pointsLength = 10;
    const x_vals = []
    const y_vals = []

    for (let i = 0; i < pointsLength; i++) {
      let rXNum = Math.round((Math.random() * 100)) / 10
      x_vals.push(rXNum)
      let rYNum = Math.round((Math.random() * 100)) / 10
      y_vals.push(rYNum)
      console.log(`${rXNum}: ${rYNum}`)
    }

    const xs = tf.tensor1d(x_vals)
    const ys = tf.tensor1d(y_vals)

    await this.linearModel.fit(xs, ys)

    console.log('Model trained!')
  }

  linearPrediction (val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any
    this.prediction = Array.from(output.dataSync())[0]
  }

  //// LOAD PRETRAINED KERAS MODEL ////
  async loadModel() {
    this.model = await tf.loadModel('/assets/model.json');
  }

  async predict(imageData: ImageData) {

    const pred = await tf.tidy(() => {

      // Convert the canvas pixels to
      const dimensions = [1, 28, 28, 1]
      let img = tf.fromPixels(imageData, 1);
      img = img.reshape(dimensions);
      img = tf.cast(img, 'float32');

      // Make and format the predications
      const output = this.model.predict(img) as any;

      // Save predictions on the component
      this.predictions = Array.from(output.dataSync()); 
      console.log(this.predictions)
    });

  }
}
