import * as THREE from 'three';
import { tensorImagePlane, imgUrlToTensor, imagePlane, showActivationAcrossPlanes } from "./common.mjs";
import * as common from "./common.mjs";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet"
import { imagenetLabels } from "./labels"
import { Text } from 'troika-three-text'
export class NetVis {
  // when I add the ability to modify activations, I'll do it by 
  static async create(world, canvas, config) {

    const thiss = new NetVis()
    await thiss.init(world, canvas, config)
    return thiss
  }

  testysfutts() {
    const extens = tf.stack([tf.add(tf.range(0, 50176).reshape([224, 224]), 1000000), tf.add(tf.range(0, 50176).reshape([224, 224]), 2000000), tf.add(tf.range(0, 50176).reshape([224, 224]), 3000000)], 2)
    console.log("extens", extens)
    const rtens = common.tensorToArray(extens)
    console.log("rtens", rtens)
    const stime = performance.now()
    const asdf = extens.dataSync()
    console.log("datasync took", performance.now() - stime)
    common.tensorToArray(extens)

    console.log("asdf", asdf)
  }

  async init(world, canvas, config) {
    this.transparency = 0.2
    this.verticalSpacing = 0.2
    this.horizontalSpacing = 0.2
    this.canvas = canvas
    this.channelsLast = false;
    this.uniformFilterSize = 2

    this.spec = { layers: {}, zoom: 0, injected: {}, input: config.input, name: config.name, imageNames: config.imageNames, cameraLocked: true, }

    this.dirs = { models: config.models, images: config.images, deepdream: config.deepdream }
    const url = this.dirs.models + "/" + this.spec.name + "/model.json"
    this.dreamurl = this.dirs.deepdream + `/filter/${this.spec.name}/`
    this.dreamcache = {}
    this.inputcache = {}
    console.log(url)

    const model = await tf.loadLayersModel(url)
    console.log(model)
    this.world = world
    this.group = new THREE.Group()
    this.group.name = ("netvis")
    world.add(this.group)
    this.group.position.z -= 8
    this.group.position.x = 0
    this.group.position.y = 0.4

    this.activationsGroup = new THREE.Group()
    this.activationsGroup.name = "activations"
    this.group.add(this.activationsGroup)

    this.topPredictions = []
    this.outputLayers = []
    for (let layer of model.layers) {
      if (layer.name.match(/conv\d?d?$/) && !layer.name.match(/bn$/)) {
        if (!this.channelsLast && layer.dataFormat === "channelsLast") {
          this.channelsLast = true
        }
        const symTensor = layer.outboundNodes[0].outputTensors[0]
        this.outputLayers.push(symTensor)
      }
    }
    const modelspec = { inputs: model.inputs, outputs: [...this.outputLayers, model.outputs[0]] }
    this.model = tf.model(modelspec)
    this.inputShape = model.feedInputShapes[0]
    this.inputShape[0] = 1

    for (let output of this.model.outputs) {
      this.spec.layers[this.actName(output)] = { show: true, fv: false, shownFilters: [1], focusedFilter: 1, shape: output.shape, _: {} }
    }
    this.spec.focusedLayer = Object.keys(this.spec.layers)[0]

    this.widthScale = 1 / 50
    this.sideSpacing = 1.5

    this.fontSize = 0.15
    this.labelOffset = 0.3

    let side = this.inputShape[1] * this.widthScale * this.sideSpacing
    for (let li = 0; li < this.outputLayers.length; li++) {
      const output = this.outputLayers[li]
      const actShape = output.shape
      const planes = []

      const activationGroup = new THREE.Group()
      activationGroup.name = this.actName(output)
      this.activationsGroup.add(activationGroup)
      activationGroup.position.x += side
      side += actShape[1] * this.widthScale * this.sideSpacing

      const outputName = this.actName(output) // .replaceAll(/(^.+\/)|(_bn)/g, "")
      const activationLabel = this.createText(outputName)
      activationGroup.add(activationLabel)

      const filtersGroup = new THREE.Group()
      filtersGroup.name = "filters"
      activationGroup.add(filtersGroup)

      const scale = this.getScale(actShape)
      activationLabel.position.y -= scale * 0.5 + this.labelOffset

      // this.activationPlaneGroups.push(planes)
    }

    this.pixelSelectObj = new THREE.Mesh(new THREE.BoxGeometry(this.widthScale, this.widthScale, this.widthScale * 10), new THREE.MeshBasicMaterial({ color: 0x00ff00 }))
    this.group.add(this.pixelSelectObj)
    this.pixelSelectObj.position.z -= this.widthScale

    this.updating = false
    this.activationTensors = {}
    this.delay = 4
    this.lastUpdate = -9999999999
    // await Promise.all(this.spec.imageNames.map((x) => this.getImageTensor(x)))
    this.inputTensor = await this.getImageTensor(this.spec.imageNames[this.spec.input])
    // console.log("it", this.inputTensor.dataSync())
    this.selectedActivationIndex = 0
    this.selectedPlaneIndex = 0
    this.selectedPixel = [0, 0]

    this.inputPlane = await tensorImagePlane(this.inputTensor.squeeze(0), true)
    this.inputPlane.scale.x = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.scale.y = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.scale.z = this.inputShape[1] * this.widthScale * 0.5
    this.inputPlane.position.z = 0
    // this.inputPlane.position.y = -3
    this.inputPlane.position.y = -3
    this.group.add(this.inputPlane)

    this.setupListeners()

    console.log(this.spec)

    return this
  }

  getScale(shape) {
    return this.uniformFilterSize || shape[1] * this.widthScale
  }

  actName(x) {
    return x.name.match(/^[^/]+/)[0].replace("_bn", "_conv")
    // return x.name.match(/^[^/]+/)[0]
  }

  async getDream(layerName, idx) {
    const cachename = layerName + "." + idx
    const url = this.dreamurl + layerName + "/" + idx + ".png"
    if (this.dreamcache[cachename] !== undefined) {
      return this.dreamcache[cachename]
    } else {
      const plane = await new Promise(resolve => imagePlane(url, resolve))
      if (plane) {
        plane.name = "dream"
        plane.filter = idx
      }
      this.dreamcache[cachename] = plane
      return plane
    }
  }

  createFilterVisual(symbolicTensor, idx) {
    const shape = symbolicTensor.shape
    const plane = tensorImagePlane()
  }

  async getImageTensor(name) {
    if (this.inputcache[name] !== undefined) {
      return tf.add(this.inputcache[name], 0)
    }
    const url = this.dirs.images + "/" + name
    const t = await imgUrlToTensor(url)
    const result = t
    this.inputcache[name] = result
    return tf.add(result, 0)
  }

  //@STUCK I don't know of a way to alter the middle of a compute graph in tfjs
  async createActivationInjectedVis(activationIndex) {
    // FOR NOW THIS CORRUPTS OLD MODEL
    const thiss = new NetVis()
    const oldOutputs = this.model.outputs
    const newInputTensor = oldOutputs[activationIndex].outboundNodes[0].outputTensors[0]
    thiss.model = tf.model({ inputs: [...this.model.inputs, newInputTensor], outputs: oldOutputs.slice(activationIndex) })
    return thiss
  }

  getFeatureVisualizationPlane(name, number) {
    const url = `./deepdream/filter/${this.model.name}/${name}/${number}.jpg`
    const plane = imagePlane(url)
    return plane
  }

  translateSelectedPixel(dx, dy) {
    this.selectedPixel[0] = Math.min(Math.max(this.selectedPixel[0] + dx, 0), this.inputShape[1])
    this.selectedPixel[1] = Math.min(Math.max(this.selectedPixel[1] + dy, 0), this.inputShape[2])

  }

  createText(text, size = 1) {
    const result = new Text()
    result.text = text
    result.fontSize = this.fontSize * size
    result.color = 0xFFFFFF
    const approxLength = text.length * this.fontSize * size
    result.position.x -= approxLength * 0.3
    result.name = "label"
    result.sync()
    return result
  }

  async display() {
    this.displaying = true
    const layersGroup = this.group.getObjectByName('activations')
    let xposition = 0;
    let skipped = false
    common.showActivationPlaneRGB(this.inputTensor.squeeze(0), this.inputPlane)
    for (let layerName in this.spec.layers) {
      if (layerName === "predictions") continue
      if (!skipped) {
        skipped = true;
        // continue
      }
      const layer = this.spec.layers[layerName]
      if (layer.show) {
        const layerGroup = layersGroup.getObjectByName(layerName)
        const filtersGroup = layerGroup.getObjectByName("filters")
        const output = this.activationTensors[layerName]

        layerGroup.visible = true;
        const layerShape = common.debatchShape(layer.shape)
        const scale = this.getScale(layer.shape)
        const xtaken = scale + scale + this.horizontalSpacing * 2
        layerGroup.position.x = xposition
        xposition += xtaken
        for (let i = filtersGroup.children.length; i < layer.shownFilters.length; i++) {
          const filterGroup = new THREE.Group()
          const label = this.createText("i", 2)
          filterGroup.add(label)
          label.position.x -= 0.01
          label.position.y += this.fontSize * 1

          filtersGroup.add(filterGroup)
          const plane = await tensorImagePlane(tf.zeros(layerShape))
          filterGroup.add(plane)
          plane.name = "filter"
          plane.position.z += i * 0.05
          plane.scale.x = scale
          plane.scale.z = scale
          plane.scale.y = scale
          plane.position.x = -(scale + this.horizontalSpacing) / 2
        }
        for (let i = layer.shownFilters.length; i < filtersGroup.children.length; i++) {
          const plane = filtersGroup.children[i]
          plane.visible = false
        }
        const offset = layer.shownFilters.findIndex(x => x === layer.focusedFilter)
        const height = scale + this.verticalSpacing
        let zposition = -height * offset
        for (let i = 0; i < layer.shownFilters.length; i++) {
          const filter = layer.shownFilters[i]
          const filterGroup = filtersGroup.children[i]
          const filterLabel = filterGroup.getObjectByName("label")
          filterLabel.text = filter
          filterLabel.sync()
          const filterPlane = filterGroup.getObjectByName("filter")
          const filterTensor = tf.slice(output, [0, 0, 0, filter], [-1, -1, -1, 1]).squeeze(0)
          filterGroup.position.z = zposition
          zposition += height
          // console.log(filterTensor.dataSync())
          common.showActivationPlane(filterTensor, filterPlane)
          const oldDream = filterGroup.getObjectByName("dream")
          if (oldDream && oldDream.filter !== filter) {
            filterGroup.remove(oldDream)
          }
          if (!oldDream || oldDream.filter !== filter) {
            this.getDream(layerName, filter).then(dream => {
              if (dream) {
                filterGroup.add(dream)
                dream.position.x = (scale + this.horizontalSpacing) / 2
                if (this.uniformFilterSize) {
                  dream.scale.x = scale
                  dream.scale.y = scale
                  dream.scale.z = scale
                }
              }
            })
          }
          if (this.spec.cameraLocked && layer.focusedFilter == filter) {
            this.world.position.z = -zposition
          }
        }
        if (this.spec.cameraLocked && this.spec.focusedLayer == layerName) {
          this.world.position.x = -xposition + 4
          this.inputPlane.position.x = xposition - 4 - 0.5
        }
      } else {
        layerGroup.visible = false;
      }
    }
    if (this.spec.cameraLocked) {
      this.world.position.z = this.spec.zoom * 2
    }
    this.displaying = false
  }

  async _update() {
    const ustime = performance.now()

    for (let k in this.activationTensors) {
      const t = this.activationTensors[k]
      if (t) t.dispose()
    }
    this.activationTensors = {}
    const pstime = performance.now()
    console.log("inputtensor")
    this.inputTensor.print()
    common.tfMode()

    const at = this.model.predict(this.inputTensor)
    for (let i = 0; i < this.model.outputs.length; i++) {
      const output = this.model.outputs[i]
      this.activationTensors[this.actName(output)] = at[i]
    }
    console.log(this.activationTensors)
    this.probs = at[this.model.outputs.length - 1]
    at[1].print()
    const dstime = performance.now()
    this.probs.data().then(probsArray => {
      // const arr = common.tensorToArray(this.probs)
      console.log('data took', performance.now() - dstime)
      const zipped = []
      for (let i = 0; i < probsArray.length; i++) {
        zipped.push([probsArray[i], i])
      }
      zipped.sort((a, b) => b[0] - a[0])

      for (let i = 0; i < 1; i++) {
        console.log(imagenetLabels[zipped[i][1]])
      }
    })
    console.log(`predict took ${performance.now() - pstime}`)

    await this.display()
    console.log("took", performance.now() - ustime)
  }

  update(inputs) {
    this.userInputs = inputs
    if (!this.updating && ((this.lastUpdate + this.delay * 1000 < performance.now()) || this.activationsDirty)) {
      this.updating = true
      this.lastUpdate = performance.now()
      this._update().then(() => {
        this.updating = false
        this.activationsDirty = false
      })
    } else if (this.visualDirty) {
      this.display()
      this.visualDirty = false
    }
    return {}
  }

  selectNextLayerOrder(t) {
    const keys = Object.keys(this.spec.layers)
    const oldIndex = keys.indexOf(this.spec.focusedLayer)
    const newIndex = Math.min(Math.max(oldIndex + t, 0), keys.length - 1)
    const newKey = keys[newIndex]
    this.spec.focusedLayer = newKey
    this.setToDisplay()
  }

  selectNextFilterOrder(t) {
    const layer = this.spec.layers[this.spec.focusedLayer]
    const curIndex = layer.shownFilters.indexOf(layer.focusedFilter)
    const newFilter = Math.min(Math.max(layer.focusedFilter + t, 0), layer.shape[layer.shape.length - 1] - 1)
    layer.focusedFilter = newFilter
    layer.shownFilters[curIndex] = newFilter
    this.setToDisplay()
  }

  selectNextFilterList(t) {
    const layer = this.spec.layers[this.spec.focusedLayer]
    const newFilter = layer.shownFilters[Math.min(Math.max(layer.shownFilters.indexOf(layer.focusedFilter) + t, 0), layer.shownFilters.length - 1)]
    layer.focusedFilter = newFilter
    this.setToDisplay()
  }

  zoomIn() {
    this.spec.zoom += 1
    this.setToDisplay()
  }

  zoomOut() {
    this.spec.zoom -= 1
    this.setToDisplay()
  }

  setToDisplay() {
    this.visualDirty = true
  }

  setToRecalculate() {
    this.activationsDirty = true
  }

  async cycleInputs(x) {
    this.spec.input = Math.min(Math.max(this.spec.input + x, 0), this.spec.imageNames.length - 1)
    this.inputTensor = await this.getImageTensor(this.spec.imageNames[this.spec.input])
    console.log("inputTensor", this.inputTensor.dataSync())
    this.setToRecalculate()
  }

  setupListeners() {
    document.addEventListener("keydown", async (event) => {
      let caught = true
      if (event.ctrlKey) {
        switch (event.key) {
          case "ArrowRight":
          case "d":
            this.translateSelectedPixel(1, 0)
            break;
          case "ArrowLeft":
          case "a":
            this.translateSelectedPixel(-1, 0)
            break;
          case "ArrowUp":
          case "w":
            this.translateSelectedPixel(0, 1)
            break;
          case "s":
          case "ArrowDown":
            this.translateSelectedPixel(0, -1)
            break;
          default:
            caught = false;
        }
      } else {
        switch (event.key) {
          case "ArrowUp":
          case "w":
            this.selectNextFilterOrder(1)
            break;
          case "ArrowDown":
          case "s":
            this.selectNextFilterOrder(-1)
            break;
          case "ArrowRight":
          case "d":
            this.selectNextLayerOrder(1)
            break;
          case "ArrowLeft":
          case "a":
            this.selectNextLayerOrder(-1)
            break;
          case "e":
            this.zoomIn()
            break
          case "q":
            this.zoomOut()
            break
          case "n":
            console.log(this)
            await this.cycleInputs(-1)
            // await new Promise(resolve => setTimeout(resolve, 50))
            break
          case "m":
            await this.cycleInputs(1)
            break
          case "r":
            this.setToRecalculate()
            break
          case "b":
            const varsy = tf.tensor(new Float32Array(10000), [10000], 'float32')
            console.log(varsy)
            break
          case "g":
            this.getImageTensor(this.spec.imageNames[this.spec.input])
            break
          default:
            caught = false;
        }
      }
      if (caught) {
        event.preventDefault()
      }
    })
  }
}