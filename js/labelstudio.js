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
              <Label value="map"></Label>
              <Label value="line"></Label>
              <Label value="donut"></Label>
              <Label value="scatter"></Label>
          </RectangleLabels>
        </View>
  `


const render_labelstudio = function(annotations, meta, firstName, lastName) {
  let completions = [{result: []}]
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
        // firstName: 'Jaidev',
        // lastName: 'Deshpande'
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

      onUpdateCompletion: function(ls, completion) {
        $.ajax({
          url: 'pages',
          method: 'PUT',
          data: {
            page_id: chart_id,
            annotations: JSON.stringify(completion.serializeCompletion())
          },
          success: function() { window.location.href = '/' }
        })
      }
    }
  )
}
