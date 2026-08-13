from typing import Any

from neo4j import Driver

from config.settings import NEO4J_DATABASE


def _remove_none_values(
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value is not None
    }


def ingest_paper(
    driver: Driver,
    paper: dict[str, Any],
) -> None:
    paper_id = str(
        paper.get("openalex_id") or ""
    ).strip()

    if not paper_id:
        raise ValueError(
            "Paper OpenAlex ID is required."
        )

    paper_properties = _remove_none_values(
        {
            "title": paper.get("title"),
            "year": paper.get("year"),
            "publication_date": paper.get(
                "publication_date"
            ),
            "doi": paper.get("doi"),
            "type": paper.get("type"),
            "language": paper.get(
                "language"
            ),
            "cited_by_count": paper.get(
                "cited_by_count"
            ),
            "abstract": paper.get(
                "abstract"
            ),
        }
    )

    driver.execute_query(
        """
        MERGE (
            p:Paper {
                openalex_id: $paper_id
            }
        )
        SET p += $properties
        """,
        paper_id=paper_id,
        properties=paper_properties,
        database_=NEO4J_DATABASE,
    )

    author_entities = paper.get(
        "author_entities",
        [],
    )

    for author in author_entities:
        author_id = str(
            author.get("openalex_id")
            or ""
        ).strip()

        if not author_id:
            continue

        author_properties = (
            _remove_none_values(
                {
                    "name": author.get(
                        "name"
                    ),
                }
            )
        )

        driver.execute_query(
            """
            MATCH (
                p:Paper {
                    openalex_id: $paper_id
                }
            )

            MERGE (
                a:Author {
                    openalex_id: $author_id
                }
            )

            SET a += $properties

            MERGE (
                a
            )-[
                :AUTHORED
            ]->(
                p
            )
            """,
            paper_id=paper_id,
            author_id=author_id,
            properties=author_properties,
            database_=NEO4J_DATABASE,
        )

    venue_entity = paper.get(
        "venue_entity"
    )

    if isinstance(
        venue_entity,
        dict,
    ):
        venue_id = str(
            venue_entity.get(
                "openalex_id"
            )
            or ""
        ).strip()

        if venue_id:
            venue_properties = (
                _remove_none_values(
                    {
                        "name": (
                            venue_entity.get(
                                "name"
                            )
                        ),
                    }
                )
            )

            driver.execute_query(
                """
                MATCH (
                    p:Paper {
                        openalex_id: $paper_id
                    }
                )

                MERGE (
                    v:Venue {
                        openalex_id: $venue_id
                    }
                )

                SET v += $properties

                MERGE (
                    p
                )-[
                    :PUBLISHED_IN
                ]->(
                    v
                )
                """,
                paper_id=paper_id,
                venue_id=venue_id,
                properties=venue_properties,
                database_=NEO4J_DATABASE,
            )


def ingest_papers(
    driver: Driver,
    papers: list[dict[str, Any]],
) -> None:
    if not papers:
        raise ValueError(
            "Papers cannot be empty."
        )

    for paper in papers:
        ingest_paper(
            driver,
            paper,
        )