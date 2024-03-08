import time

import requests


class TestInterestsAPI:
    BASE_URL = "http://localhost:5005/interests"

    def make_request(self, user_handle):
        return requests.get(f"{self.BASE_URL}/{user_handle}")

    def test_basic_response(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"
        top_k = 15

        start_time = time.time()
        response = self.make_request(user_handle=user_handle)
        end_time = time.time()
        request_time = end_time - start_time

        assert response.status_code == 200, "Status code is not 200"

        response = response.json()

        assert response["user_handle"] == user_handle, "Incorrect user handle"

        assert isinstance(response["interests"], list), "Interests is not a list"

        assert len(response["interests"]) == top_k, "Incorrect number of interests"

        assert request_time < 1, "Request greater than 1 seconds"

    def test_with_probability_threshold(self):
        user_handle = "e337a675-46f5-437e-aa72-5d43643b5461"

        response = self.make_request(user_handle=user_handle)

        response = response.json()

        for interest in response["interests"]:
            assert "id" in interest, f"Interest object missing 'id': {interest}"
            assert "label" in interest, f"Interest object missing 'label': {interest}"
            assert (
                "probability" in interest
            ), f"Interest object missing 'label': {interest}"
