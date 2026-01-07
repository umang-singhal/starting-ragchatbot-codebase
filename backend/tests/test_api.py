"""
API endpoint tests for the RAG system.

Tests the FastAPI endpoints for proper request/response handling.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.api
class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_endpoint_returns_healthy_status(self, client):
        """Test that the root endpoint returns a healthy status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data

    def test_root_endpoint_returns_json(self, client):
        """Test that the root endpoint returns JSON content."""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for the /api/query endpoint."""

    def test_query_with_valid_request_returns_response(self, client, mock_rag_system):
        """Test that a valid query request returns a proper response."""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)

        # Verify the mock was called
        mock_rag_system.query.assert_called_once()

    def test_query_with_session_id_uses_provided_session(self, client, mock_rag_system):
        """Test that providing a session_id uses the existing session."""
        session_id = "existing_session_123"
        response = client.post(
            "/api/query",
            json={
                "query": "What is deep learning?",
                "session_id": session_id
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_query_without_session_id_creates_new_session(self, client, mock_rag_system):
        """Test that omitting session_id creates a new session."""
        response = client.post(
            "/api/query",
            json={"query": "Tell me about neural networks"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new_session_456"

        # Verify session manager was called to create session
        mock_rag_system.session_manager.create_session.assert_called()

    def test_query_missing_query_field_returns_422(self, client):
        """Test that missing the query field returns a validation error."""
        response = client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )

        assert response.status_code == 422

    def test_query_empty_query_accepted(self, client):
        """Test that an empty query is accepted (Pydantic doesn't enforce non-empty by default)."""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string is valid per the schema; RAG system handles it
        assert response.status_code == 200

    def test_query_returns_sources_with_links(self, client, mock_rag_system):
        """Test that sources are returned with proper structure."""
        response = client.post(
            "/api/query",
            json={"query": "What are transformers?"}
        )

        assert response.status_code == 200
        data = response.json()
        sources = data["sources"]
        assert len(sources) > 0
        assert "name" in sources[0]
        assert "link" in sources[0]

    def test_query_handles_rag_system_error(self, client, mock_rag_system):
        """Test that RAG system errors are properly handled."""
        # Make the mock raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")

        response = client.post(
            "/api/query",
            json={"query": "This will cause an error"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint."""

    def test_courses_endpoint_returns_stats(self, client, mock_rag_system):
        """Test that the courses endpoint returns course statistics."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)

        # Verify the mock was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_returns_correct_total(self, client, mock_rag_system):
        """Test that total_courses is correctly reported."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 2

    def test_courses_returns_correct_titles(self, client, mock_rag_system):
        """Test that course_titles list is correctly reported."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "Machine Learning Basics" in data["course_titles"]
        assert "Deep Learning Advanced" in data["course_titles"]

    def test_courses_handles_empty_analytics(self, client, mock_rag_system):
        """Test that empty analytics are handled correctly."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_handles_rag_system_error(self, client, mock_rag_system):
        """Test that RAG system errors are properly handled."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestSessionEndpoint:
    """Tests for the /api/session/new endpoint."""

    def test_create_session_returns_session_id(self, client, mock_rag_system):
        """Test that creating a session returns a session_id."""
        response = client.post("/api/session/new")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_create_session_calls_session_manager(self, client, mock_rag_system):
        """Test that the endpoint calls the session manager."""
        response = client.post("/api/session/new")

        assert response.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_create_session_returns_correct_session_id(self, client, mock_rag_system):
        """Test that the returned session_id matches the created one."""
        response = client.post("/api/session/new")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new_session_456"

    def test_create_session_handles_error(self, client, mock_rag_system):
        """Test that session creation errors are properly handled."""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session error")

        response = client.post("/api/session/new")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestCorsHeaders:
    """Tests for CORS middleware configuration."""

    def test_options_request_returns_cors_headers(self, client):
        """Test that OPTIONS requests return proper CORS headers."""
        response = client.options(
            "/api/query",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST"
            }
        )

        # Should succeed (CORS preflight)
        assert response.status_code == 200

    def test_query_from_different_origin_accepted(self, client):
        """Test that requests from different origins are accepted."""
        response = client.post(
            "/api/query",
            json={"query": "Test query"},
            headers={"Origin": "https://example.com"}
        )

        assert response.status_code == 200
