# Matching Tabular Data to Knowledge Graph Based on Entity Disambiguation with Table Context

[TOC]

## Overview

Tabular data to knowledge graph matching (TDKGM) aims to assign semantic tags from knowledge graphs (KGs) to the elements of the tables, including three tasks: Column Type Annotation (CTA), Cell Entity Annotation (CEA), and Columns Property Annotation (CPA). It is a non-trivial task due to missing, incomplete, or ambiguous metadata, which makes entity disambiguation more difficult.

Previous approaches mostly are based on two representative paradigms: heuristic-based and deep learning-based methods. However, the former is less robust when tackling real-world Web tables, while the latter requires much training data and time.

Consequently, we conceive the idea of introducing table context semantics and propose a novel annotation method KGCODE-Tab. First, we preprocess the tabular data via table structure analysis, spell correction, and entity recall. Then, we assign scores to the candidate entities of the cells, based on the similarities between table cells and property values in KGs. After that, we determine the subject column and find property-based matches. Finally, we complete three semantic annotation tasks based on scores without the help of non-table information (e.g., table headers and table names).

Experimental results on public datasets demonstrate that although we do not take any complicated technical route, KGCODE-Tab can disambiguate most entity mentions and significantly outperform most of the baseline methods.

## Requirements

- Python 3.11 (Typing required)
- SQLite (Database management)
- Spacy (optional, for NER)

## Quick Start

Load dataset. You can change `LimayeDataset` to others like `T2DDataset`, `MusicBrainzDataset`, `ImdbDataset`, and `SemTabDataset`. This will automatically create gt.parquet file to accelerate reading.

```py
from src.datasets import LimayeDataset
ds = LimayeDataset("datasets/Limaye", limit=50)
```

Create table processor and load cached processed tables (If no table cache, the file will be created). MessagePack format is preferred for smaller size, but json is also OK.

```py
tp = TableProcessor(".cache/limaye.msgpack")
```

Add dataset to processor. This will check whether tables in the datasets are cached. If not, tables are created and cached. Set `force` to True and force recreating.

```py
tp.add_dataset(ds, force=False)
```

Load entity cache to `EntityManager`. This is optional, but it will accelerate entity retrieval from databases.

```py
EntityManager.load(".cache/limaye-entities.msgpack")
```

Create answerer, and annotate all tables loaded to the table processor.

```py
ans = Answerer(ds)
aa = annotate_all(tp)
```

Dump entity cache for next use.

```py
EntityManager.dump(".cache/limaye-entities.msgpack")
```

Fill the answer sheet and evaluate by evaluators provided. Feel free to add any information to metadata.

```py
ans.answer(aa, ".result", [CEA_Evaluator(ds, True)], metadata={"method": "..."})
```

Generate report file which visualizes historical results. It will open it with browser by default.

```py
generate_report(".result")
```

The report will be like below:
![demo-report](img/demo-report.png)

We also have profiler support at hand. Just surround your main function with `Profiler`. It used CProfile and SnakeVi

```py
if __name__ == "__main__":
    with Profile():
        main()
```

## Folder Structure

```dir
|-src
| |-analysis      // For disambiguation and annotation, including scoring and property match.
| |-datasets      // Universal dataset adapters for processing and evaluating.
| |-evaluators    // Answer tasks with annotation results, perform evaluation and generate HTML report.
| |-process       // Preprocess and databases. Including classes for spell correction and wikidata search.
| |-searchmanage  // Multithread searcher implementation.
| |-table         // ORM classes for representing table and entity.
| |-utils         // Utilities used in preprocess and disambiguation.
|-tests           // Code for test.
```

## Supplemental Material Statement

Source code and constructed datasets are available in the supplemental material and will be released on GitHub.

## License
