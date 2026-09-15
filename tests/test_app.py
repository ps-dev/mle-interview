import time

import pytest
import requests

HTTP_OK = 200


@pytest.mark.usefixtures("_servers_running")
class TestInterestsAPI:
    BASE_URL = "http://localhost:5005/interests"

    def make_request(self, user_handle, **params):
        return requests.get(f"{self.BASE_URL}/{user_handle}", params=params)

    def test_basic_response(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"

        response = self.make_request(user_handle=user_handle)

        assert response.status_code == HTTP_OK, "Status code is not 200"

        response = response.json()

        assert response["user_handle"] == user_handle, "Incorrect user handle"

        assert isinstance(response["interests"], list), "Interests is not a list"

    def test_interests_count(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"
        top_k = 15

        response = self.make_request(user_handle=user_handle)

        assert response.status_code == HTTP_OK, "Status code is not 200"

        response = response.json()

        assert len(response["interests"]) == top_k, "Incorrect number of interests"

    def test_response_time(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"

        start_time = time.time()
        response = self.make_request(user_handle=user_handle)
        end_time = time.time()
        request_time = end_time - start_time

        assert response.status_code == HTTP_OK, "Status code is not 200"

        assert request_time < 1, "Request greater than 1 second"

    def test_with_probability_threshold(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"

        response = self.make_request(user_handle=user_handle)

        response = response.json()

        for interest in response["interests"]:
            assert "id" in interest, f"Interest object missing 'id': {interest}"
            assert "label" in interest, f"Interest object missing 'label': {interest}"
            assert (
                "probability" in interest
            ), f"Interest object missing 'probability': {interest}"
