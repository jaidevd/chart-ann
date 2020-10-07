# Chart annotation

## Endpoints

### Home page - /

Paste chart image and set a label against it.

### Label validation - /validate

For a labeled chart, one can update its label if it's annotated incorrectly.

### Validated labels - /validated

Show original chart label, its updated label and status (if validated).

### Gallery - /gallery

Show chart and its original label.

## Setup

### Dependencies
- needs `charts.db`, a sqlite file with a `charts` table
  - columns: `label`, `validated_label`, `image`, `chart_id`, `is_validated`

### Instructions

- [Install Gramex 1.x](https://learn.gramener.com/guide/install/)
- Clone this repository
- Copy assets from shared repo, e.g. `demo.gramener.com:/deploy/<user>/<repo>/`
- From the repo folder, run `gramex setup .`
- From the repo folder, run `gramex`

## Contributions

- Jaidev Deshpande <deshpande.jaidev@gmail.com>
- Bhanu Kamapantula <talk2kish@gmail.com>
