import json
from dotenv import load_dotenv
from pydantic import Field
import numpy as np
from enum import Enum
from typing import Tuple

import pandas as pd
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

import matplotlib.pyplot as plt
import os

class Direction(str, Enum):  
    """Enum to classify trend direction."""
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"
    UP = "UP"

class DirectionBM(BaseModel):
    """Pydantic model for direction classification."""
    direction: Direction = Field(..., description="The overall trend direction (DOWN, NEUTRAL, or UP)")

class Prompter:
    """Handles prompt generation, LLM invocation, and response parsing."""
    
    def __init__(self, model:str="gpt-4o-mini", temperature:float=0.7):
        self.client = self._load_env()
        self.llm = ChatOpenAI(temperature=temperature, model=model)
        self.direction_parser = PydanticOutputParser(pydantic_object=DirectionBM)
        self.generate_direction_prompt = self._create_direction_prompt()

    def _load_env(self):
        """Loads OpenAI API key from .env"""
        load_dotenv("./resources/.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found.")
        print("API Key loaded successfully.")
        return api_key

    def _create_direction_prompt(self):
        """Creates the prompt template for analyzing tabular data direction."""
        return PromptTemplate(
            template="Analyze the following tabular data (delimited by triple backticks) "
                     "and determine the overall trend direction.\n\n"
                     "Possible directions:\n"
                     "- DOWN: If the trend is decreasing overall.\n"
                     "- NEUTRAL: If there is no clear increasing or decreasing trend.\n"
                     "- UP: If the trend is increasing overall.\n\n"
                     "DATA:\n```\n{data}\n```\n\n"
                     "Provide the result as a JSON object following this format:\n{format_instructions}",
            input_variables=["data"],  
            partial_variables={"format_instructions": self.direction_parser.get_format_instructions()}
        )

    def _parse_response(self, response, model):
        """Parses the LLM response using the expected Pydantic model."""
        try:
            # Clean response: Remove triple backticks and 'json' keyword
            cleaned_response = response.content.strip().strip("```json").strip("```").strip()
            
            # Debugging: Print cleaned response
            print(f"Cleaned Response: {cleaned_response}")

            # Parse as JSON first to ensure correctness
            parsed_json = json.loads(cleaned_response)

            # Pass JSON to Pydantic for validation
            return model.parse_obj(parsed_json)
        except Exception as e:
            print(ValueError(f"Failed to parse response: {e}"))
            return None

        
    def analyze_trend(self, data_str: str, save_prompt=False):
        """Generates a prompt, invokes the LLM, and parses the response."""
        # Render the final prompt with actual data
        final_prompt = self.generate_direction_prompt.format(data=data_str)
        
        # Save prompt if needed
        if save_prompt:
            os.makedirs("./data/prompts", exist_ok=True)
            with open("./data/prompts/direction_prompt.txt", "w", encoding="utf-8") as f:
                f.write(final_prompt)
        
        # Invoke the LLM with the formatted prompt
        response = self.llm.invoke(final_prompt)
        
        return self._parse_response(response, DirectionBM)


class TSDataGenerator:
    def __init__(self, low_thresh=-0.2, high_thresh=0.2, num_points=100):
        """
        Initialize the Time Series Data Generator with classification thresholds.

        Parameters:
        - low_thresh: Lower threshold for classifying a negative trend.
        - high_thresh: Upper threshold for classifying a positive trend.
        - num_points: Number of data points in the generated series.
        """
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.num_points = num_points
        self.x_vec = None
        self.y_vec = None
        self.time_vec = None
        self.data_str = None
        self.label = None

    def _calculate_slope(self, x_vec, y_vec):
        """Calculate the slope of a given set of x and y values."""
        return np.polyfit(x_vec, y_vec, 1)[0]

    def _classify_graph(self, x_vec, y_vec) -> Direction:
        calculated_slope = self._calculate_slope(x_vec, y_vec)
        if calculated_slope > self.high_thresh:
            return Direction.UP
        elif calculated_slope < self.low_thresh:
            return Direction.DOWN
        else:
            return Direction.NEUTRAL
 
    def _create_data_str(self, time_vec, y_vec):
        """Create a tabular data string from the time series data."""
        data_str = "Date,Value\n"
        for i in range(len(time_vec)):
            data_str += f"{time_vec[i]}, {y_vec[i]}\n"
        return data_str
    
    def _gen_graph(self, param_dict, save_name, label, x_vec, y_vec):
        """Generate a graph based on the x and y values."""

        plt.plot(x_vec, y_vec)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.text(x_vec.mean(), y_vec.max() * 1.05, label, fontsize=12, ha='center', color='red')
        plt.title(save_name)
        os.makedirs('./figures', exist_ok=True)
        plt.savefig(f'./figures/{save_name}.png')

    def create_time_series_data(
            self, slope: float, intercept: float, noise: float, start: str, end: str, 
            num_points: int, gen_graph = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Generate a CSV-formatted time series dataset with timestamps and values.
        
        Parameters:
        - slope: The slope of the linear trend.
        - intercept: The y-intercept.
        - noise: Standard deviation of noise.
        - start: Start time (string in ISO format or datetime).
        - end: End time (string in ISO format or datetime).
        - num_points: Number of data points.
        - gen_graph: Whether to generate a graph.

        Returns:
        - x_vec: Array of x values.
        - y_vec: Array of y values.
        - time_vec: Array of timestamps.
        - data_str: CSV-formatted data string.
        - label: Trend classification label.
        """
        # Generate normalized x values
        x_vec = np.linspace(0, 1, num_points)  

        # Generate y values with trend + noise
        y_vec = slope * x_vec + intercept + (noise * np.random.randn(num_points))


        # Convert start and end to datetime if they are strings
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        # Generate evenly spaced time vector
        time_vec = pd.date_range(start=start_dt, end=end_dt, periods=num_points)
        
        # Convert to formatted string array
        time_vec = time_vec.astype(str)
        time_vec.to_numpy()

        # Classify the trend
        label = self._classify_graph(x_vec, y_vec)

        # Create tabular data string
        data_str = self._create_data_str(time_vec, y_vec)

        # Generate graph if specified
        
        save_name = f"slope_{slope}_intercept_{intercept}_noise_{noise}.png"
        if gen_graph:
            param_dict = {
                'slope': slope,
                'intercept': intercept,
                'noise': noise
            }
            self._gen_graph(param_dict, save_name, label, x_vec, y_vec)

        self.x_vec = x_vec
        self.y_vec = y_vec
        self.time_vec = time_vec
        self.data_str = data_str
        self.label = label