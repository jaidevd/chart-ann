const interfaces = [
    "panel",
    "update",
    "controls",
    "side-column",
    "completions:menu",
    "completions:add-new",
    "completions:delete",
    "predictions:menu",
  ]
const lsConfig =
  `
          <View>
            <Image name="img" value="$image"></Image>
            <RectangleLabels name="tag" toName="img">
              <Label value="circlepacking"></Label>
              <Label value="barchart"></Label>
              <Label value="waterfall"></Label>
              <Label value="network"></Label>
              <Label value="treemap"></Label>
              <Label value="chord"></Label>
              <Label value="map"></Label>
              <Label value="dotmatrix"></Label>
              <Label value="bubblechartlegend"></Label>
              <Label value="funnel"></Label>
              <Label value="heatmap"></Label>
              <Label value="linechart"></Label>
              <Label value="proportional-area-chart"></Label>
              <Label value="nightingalerose"></Label>
              <Label value="radialbar"></Label>
              <Label value="dendrogram"></Label>
              <Label value="fish bone chart"></Label>
              <Label value="sunburst"></Label>
              <Label value="donut"></Label>
              <Label value="area-chart"></Label>
              <Label value="table"></Label>
              <Label value="boxplot"></Label>
              <Label value="wordcloud"></Label>
              <Label value="radarchart"></Label>
              <Label value="bubblecloud"></Label>
              <Label value="pyramid"></Label>
              <Label value="parliament"></Label>
              <Label value="sankey"></Label>
              <Label value="parallelcoordinates"></Label>
              <Label value="bullet"></Label>
              <Label value="ranking"></Label>
              <Label value="butterfly"></Label>
              <Label value="bump"></Label>
              <Label value="nestedcirclepack"></Label>
              <Label value="bubblematrix"></Label>
              <Label value="pictogram"></Label>
              <Label value="bartime"></Label>
              <Label value="marimekko"></Label>
              <Label value="swarm-plot"></Label>
              <Label value="circularbar"></Label>
          </RectangleLabels>
        </View>
  `

const render_labelstudio = function(annotations, meta, firstName, lastName) {
  var completions = [{result: []}]
  _.each(annotations, function(item) {completions[0].result.push(item)})
  let chart_id = meta.inserted[0].page_id
  return new LabelStudio('label-studio',
    {
      config: lsConfig,
      interfaces: interfaces,
      user: {
        pk: 1,
        firstName: firstName,
        lastName: lastName
      },

      task: {
        completions: completions,
        predictions: [],
        id: 1,
        data: {
          image: `../urlann/${chart_id}`
        }
      },

      onLabelStudioLoad: function(LS) {
        var c = LS.completionStore.addCompletion({
          userGenerate: true
        });
        LS.completionStore.selectCompletion(c.id);
      },

      // onSubmitCompletion: function(ls, completion) {
      //   $.ajax({
      //     url: `../updateLabel/${chart_id}`,
      //     method: 'PUT',
      //     data: JSON.stringify(completion.serializeCompletion()),
      //     processData: false
      //   })
      // },
      // onUpdateCompletion: function(ls, completion) {
      //   $.ajax({
      //     url: `../updateLabel/${chart_id}`,
      //     method: 'PUT',
      //     data: JSON.stringify(completion.serializeCompletion()),
      //     processData: false
      //   })
      // }
    }
  )
}
