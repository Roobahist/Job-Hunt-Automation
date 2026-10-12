from urllib.parse import parse_qs, urlsplit

from job_hunt.application.discovery import build_search_url


def test_build_search_url_matches_previous_n8n_filters_except_date_source() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {
            "Generated URL": "https://stale.example/?keywords=wrong&f_TPR=r999",
            "Keywords": "software engineer",
            "Location": "canada",
            "Experience Level": [
                {"value": "Entry Level"},
                {"value": "Internship"},
                {"value": "Associate"},
            ],
            "Job Type": [{"value": "Full-time"}, {"value": "Contract"}],
            "Workplace Type": [{"value": "Remote"}, {"value": "Hybrid"}],
            "Date Posted": "month",
            "Sort By": {"value": "Date"},
            "Distance": 25,
            "Company ID": "12345,67890",
            "Easy Apply": True,
        },
        2,
    )
    params = parse_qs(urlsplit(url).query)

    assert params == {
        "keywords": ["software engineer"],
        "location": ["canada"],
        "f_E": ["2,1,3"],
        "f_JT": ["F,C"],
        "f_WT": ["2,3"],
        "f_TPR": ["r7200"],
        "sortBy": ["DD"],
        "distance": ["25"],
        "f_C": ["12345,67890"],
        "f_AL": ["true"],
    }
    assert "stale.example" not in url
    assert "r999" not in url


def test_build_search_url_uses_config_interval_not_date_posted() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {"Keywords": "data scientist", "Location": "alberta", "Date Posted": "month"},
        1,
    )
    assert url == (
        "https://www.linkedin.com/jobs/search/?keywords=data+scientist&location=alberta&f_TPR=r3600&sortBy=R"
    )


def test_build_search_url_omits_optional_filters_when_not_selected() -> None:
    url = build_search_url(
        "https://www.linkedin.com/jobs/search/",
        {"Keywords": "machine learning", "Easy Apply": False},
        1.5,
    )
    params = parse_qs(urlsplit(url).query)
    assert params["f_TPR"] == ["r5400"]
    assert params["sortBy"] == ["R"]
    assert "f_AL" not in params
