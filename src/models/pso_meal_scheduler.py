# src/models/pso_meal_scheduler.py (UPDATED VERSION)
import numpy as np
import random
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.portion_adjuster import get_user_nutrition_targets
from src.models.user_preferences import UserPreferences
from src.models.cbf_recommender import CBFRecommender

try:
    from src.utils.mlflow_manager import start_run, log_params, log_metrics, log_artifact, set_tag

    MLFLOW_AVAILABLE = True
except ImportError:
    # Fallback jika mlflow_manager tidak tersedia
    MLFLOW_AVAILABLE = False


    def start_run(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()


    def log_params(*args, **kwargs):
        pass


    def log_metrics(*args, **kwargs):
        pass


    def log_artifact(*args, **kwargs):
        pass


    def set_tag(*args, **kwargs):
        pass


class ParticleSwarmOptimizer:
    def __init__(self,
                 num_particles=30,
                 num_days=7,
                 meals_per_day=3,
                 recipes_per_meal=3,
                 max_iterations=100,
                 w=0.7,
                 c1=1.5,
                 c2=1.5):
        """
        Initialize the PSO algorithm for meal scheduling

        Args:
            num_particles: Number of particles in the swarm
            num_days: Number of days to schedule (default: 7)
            meals_per_day: Number of meals per day (default: 3)
            recipes_per_meal: Number of recipes to combine per meal (default: 3)
            max_iterations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive coefficient (personal best)
            c2: Social coefficient (global best)
        """
        self.num_particles = num_particles
        self.num_days = num_days
        self.meals_per_day = meals_per_day
        self.recipes_per_meal = recipes_per_meal
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Load meal data
        self.cbf_recommender = CBFRecommender()
        self.meal_data = self.cbf_recommender.meal_data
        self.meal_ids = self.meal_data['ID'].unique().tolist()

        # Parameters for the fitness function
        self.target_metrics = {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0,
        }

        # Weights for different nutrients in fitness calculation
        self.nutrient_weights = {
            'calories': 15.0,
            'protein': 2.0,
            'fat': 8.0,
            'carbohydrates': 3.0,
            'fiber': 1.5
        }

    def set_user_requirements(self, age, gender, weight, height, activity_level,
                              goal='maintain', user_preferences=None):
        """
        Set user requirements for meal optimization dengan MLflow tracking
        """
        # MLflow tracking dengan fallback
        with start_run(run_name="Nutrition Targets", experiment_name="Meal Planning", nested=True):
            # Log user parameters
            user_params = {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "goal": goal
            }
            log_params(user_params)

            # Set the user preferences for filtering recipes
            self.user_preferences = user_preferences

            self.goal = goal

            # Filter meals based on user preferences if provided
            if self.user_preferences:
                self.filtered_meal_data = self.user_preferences.filter_recipes(self.meal_data)
                if self.filtered_meal_data.empty:
                    raise ValueError("No recipes match the user preferences.")
                self.meal_ids = self.filtered_meal_data['ID'].unique().tolist()
            else:
                self.filtered_meal_data = self.meal_data

            # Apply calorie-based filtering
            self.filter_recipes_by_calorie_target()

            # Calculate target metrics based on user parameters
            user_profile = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'activity_level': activity_level,
                'goal': goal,
                'meals_per_day': self.meals_per_day
            }
            self.target_metrics = get_user_nutrition_targets(user_profile)

            # Adaptive weights based on target calories
            target_calories = self.target_metrics['calories']

            if target_calories < 2000:
                self.nutrient_weights = {
                    'calories': 25.0,
                    'protein': 3.0,
                    'fat': 15.0,
                    'carbohydrates': 5.0,
                    'fiber': 2.0
                }
            elif target_calories > 3000:
                self.nutrient_weights = {
                    'calories': 8.0,
                    'protein': 2.0,
                    'fat': 6.0,
                    'carbohydrates': 2.0,
                    'fiber': 1.5
                }
            else:
                self.nutrient_weights = {
                    'calories': 15.0,
                    'protein': 2.0,
                    'fat': 8.0,
                    'carbohydrates': 3.0,
                    'fiber': 1.5
                }

            # Log the target metrics
            log_params(self.target_metrics)

            print(f"Daily targets set: Calories: {self.target_metrics['calories']:.0f}, "
                  f"Protein: {self.target_metrics['protein']:.1f}g, "
                  f"Fat: {self.target_metrics['fat']:.1f}g, "
                  f"Carbs: {self.target_metrics['carbohydrates']:.1f}g, "
                  f"Fiber: {self.target_metrics['fiber']:.1f}g")

    def calculate_nutritional_value(self, meal_schedule):
        """Calculate total nutritional value for a meal schedule"""
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0,
        }

        for day in meal_schedule:
            for meal in day:
                for meal_id in meal:
                    meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(meal_id)]
                    if not meal_record.empty:
                        record = meal_record.iloc[0]

                        if 'Adjusted_Total_Nutrition' in record:
                            nutrition = record['Adjusted_Total_Nutrition']
                        else:
                            nutrition = record['nutrition']

                        total_nutrition['calories'] += nutrition.get('calories', 0)
                        total_nutrition['protein'] += nutrition.get('protein', 0)
                        total_nutrition['fat'] += nutrition.get('fat', 0)
                        total_nutrition['carbohydrates'] += nutrition.get('carbohydrates', 0)
                        total_nutrition['fiber'] += nutrition.get('fiber', 0)

        # Return average per day
        days = len(meal_schedule)
        return {k: v / days for k, v in total_nutrition.items()}

    def calculate_meal_variety(self, meal_schedule):
        """Calculate the variety score for a meal schedule"""
        flat_schedule = [recipe for day in meal_schedule for meal in day for recipe in meal]
        unique_meals = len(set(flat_schedule))
        total_meals = len(flat_schedule)
        variety_score = unique_meals / total_meals if total_meals > 0 else 0
        return variety_score

    def fitness_function(self, meal_schedule):
        """Calculate fitness score with exponential penalty"""
        daily_nutrition = self.calculate_nutritional_value(meal_schedule)

        # Calculate penalty for deviation from target with exponential scaling
        penalty = 0
        for nutrient, target in self.target_metrics.items():
            if target > 0 and nutrient in daily_nutrition:
                # Calculate relative error as percentage
                relative_error = abs(daily_nutrition[nutrient] - target) / target

                if relative_error > 0.5:
                    exponential_penalty = relative_error ** 2.5
                elif relative_error > 0.2:
                    exponential_penalty = relative_error ** 2
                else:  # < 20% error gets linear penalty
                    exponential_penalty = relative_error

                # Apply weight to the penalty
                weighted_error = exponential_penalty * self.nutrient_weights.get(nutrient, 1.0)
                penalty += weighted_error

        # Calculate meal variety (higher is better)
        variety_score = self.calculate_meal_variety(meal_schedule)
        # Convert to penalty (lower is better)
        variety_penalty = 1 - variety_score

        # Combine nutrition and variety penalties
        # 85% weight to nutrition (higher than before), 15% to variety
        total_penalty = (0.85 * penalty) + (0.15 * variety_penalty)

        return total_penalty

    def initialize_swarm(self):
        """Initialize particles with random meal schedules"""
        positions = []
        for _ in range(self.num_particles):
            schedule = []
            for _ in range(self.num_days):
                day_meals = []
                for _ in range(self.meals_per_day):
                    meal_recipes = random.choices(self.meal_ids, k=self.recipes_per_meal)
                    day_meals.append(meal_recipes)
                schedule.append(day_meals)
            positions.append(schedule)

        velocities = []
        for _ in range(self.num_particles):
            v = []
            for _ in range(self.num_days):
                day_v = []
                for _ in range(self.meals_per_day):
                    meal_v = [random.uniform(0, 0.3) for _ in range(self.recipes_per_meal)]
                    day_v.append(meal_v)
                v.append(day_v)
            velocities.append(v)

        pbests = positions.copy()
        pbest_scores = [self.fitness_function(p) for p in pbests]

        gbest_idx = np.argmin(pbest_scores)
        gbest = pbests[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        return positions, velocities, pbests, pbest_scores, gbest, gbest_score

    def update_velocity(self, positions, velocities, pbests, gbest, particle_idx):
        """Update velocity for a single particle"""
        new_velocity = []
        position = positions[particle_idx]
        velocity = velocities[particle_idx]
        pbest = pbests[particle_idx]

        for day_idx in range(len(position)):
            day_v = []
            for meal_idx in range(len(position[day_idx])):
                meal_v = []
                for recipe_idx in range(len(position[day_idx][meal_idx])):
                    current_recipe = position[day_idx][meal_idx][recipe_idx]
                    current_v = velocity[day_idx][meal_idx][recipe_idx]

                    r1, r2 = random.random(), random.random()

                    cognitive = 0
                    if pbest[day_idx][meal_idx][recipe_idx] != current_recipe:
                        cognitive = 1

                    social = 0
                    if gbest[day_idx][meal_idx][recipe_idx] != current_recipe:
                        social = 1

                    new_v = (self.w * current_v) + \
                            (self.c1 * r1 * cognitive) + \
                            (self.c2 * r2 * social)

                    new_v = max(0, min(1, new_v))
                    meal_v.append(new_v)
                day_v.append(meal_v)
            new_velocity.append(day_v)

        return new_velocity

    def update_position(self, position, velocity):
        """Update position based on velocity"""
        new_position = []

        for day_idx, day_meals in enumerate(position):
            new_day = []
            for meal_idx, meal_recipes in enumerate(day_meals):
                new_meal = []
                for recipe_idx, recipe_id in enumerate(meal_recipes):
                    if random.random() < velocity[day_idx][meal_idx][recipe_idx]:
                        new_recipe = random.choice(self.meal_ids)
                        while new_recipe == recipe_id:
                            new_recipe = random.choice(self.meal_ids)
                        new_meal.append(new_recipe)
                    else:
                        new_meal.append(recipe_id)
                new_day.append(new_meal)
            new_position.append(new_day)

        return new_position

    def optimize(self):
        """Run the PSO algorithm dengan MLflow tracking"""
        with start_run(run_name="PSO Optimization", experiment_name="Meal Planning", nested=True) as run:
            # Log PSO parameters
            pso_params = {
                "num_particles": self.num_particles,
                "max_iterations": self.max_iterations,
                "inertia": self.w,
                "cognitive": self.c1,
                "social": self.c2,
                "meals_per_day": self.meals_per_day,
                "recipes_per_meal": self.recipes_per_meal
            }
            log_params(pso_params)

            # Initialize swarm
            positions, velocities, pbests, pbest_scores, gbest, gbest_score = self.initialize_swarm()

            # Optimization loop
            for iteration in range(self.max_iterations):
                for i in range(self.num_particles):
                    velocities[i] = self.update_velocity(positions, velocities, pbests, gbest, i)
                    positions[i] = self.update_position(positions[i], velocities[i])
                    fitness = self.fitness_function(positions[i])

                    if fitness < pbest_scores[i]:
                        pbests[i] = positions[i].copy()
                        pbest_scores[i] = fitness

                        if fitness < gbest_score:
                            gbest = positions[i].copy()
                            gbest_score = fitness

                # Progress logging
                if (iteration + 1) % 10 == 0 or iteration == 0:
                    avg_nutrition = self.calculate_nutritional_value(gbest)
                    print(f"Iteration {iteration + 1}: Best fitness = {gbest_score:.4f}, "
                          f"Calories = {avg_nutrition['calories']:.0f}, "
                          f"Protein = {avg_nutrition['protein']:.1f}g")

                # MLflow metrics logging
                if iteration % 5 == 0:
                    current_nutrition = self.calculate_nutritional_value(gbest)
                    metrics = {
                        "fitness": gbest_score,
                        "calories": current_nutrition['calories'],
                        "protein": current_nutrition['protein'],
                        "variety": self.calculate_meal_variety(gbest)
                    }
                    log_metrics(metrics, step=iteration)

            # Final metrics
            nutrition = self.calculate_nutritional_value(gbest)
            final_metrics = {
                "final_fitness": gbest_score,
                "final_calories": nutrition['calories'],
                "final_protein": nutrition['protein'],
                "final_fat": nutrition['fat'],
                "final_carbs": nutrition['carbohydrates'],
                "final_fiber": nutrition['fiber'],
                "unique_meals": self.calculate_meal_variety(gbest)
            }
            log_metrics(final_metrics)

            # Log artifacts (hanya jika MLflow tersedia)
            try:
                temp_plan_path = "temp_best_plan.json"
                with open(temp_plan_path, 'w') as f:
                    json.dump(gbest, f)
                log_artifact(temp_plan_path, "meal_plans")
                if os.path.exists(temp_plan_path):
                    os.remove(temp_plan_path)
            except Exception as e:
                print(f"Warning: Could not save artifact: {e}")

            return gbest, nutrition, gbest_score

    def filter_recipes_by_calorie_target(self):
        """
        Expo-ready calorie filtering with goal-based adjustments
        """
        target_calories = self.target_metrics['calories']
        goal = getattr(self, 'goal', 'maintain')  # Get goal from user preferences

        total_recipes_per_day = self.meals_per_day * self.recipes_per_meal
        base_target_per_recipe = target_calories / total_recipes_per_day

        if goal == 'lose':
            adjustment_factor = 1.0
            max_multiplier = 1.8
        elif goal == 'gain':
            adjustment_factor = 1.0
            max_multiplier = 2.0
        else:
            adjustment_factor = 0.95
            max_multiplier = 1.6

        target_per_recipe = base_target_per_recipe * adjustment_factor

        print(f"Goal: {goal}, Target calories: {target_calories:.0f} kcal")
        print(f"Adjustment factor: {adjustment_factor}, Target per recipe: {target_per_recipe:.0f} kcal")

        if target_calories < 1800:  # Low calorie targets
            min_cal = max(50, int(target_per_recipe * 0.6))
            max_cal = int(target_per_recipe * max_multiplier)
            category = "Low Calorie"

        elif target_calories < 2400:  # Medium calorie targets
            min_cal = max(80, int(target_per_recipe * 0.7))
            max_cal = int(target_per_recipe * max_multiplier)
            category = "Medium Calorie"

        elif target_calories < 3000:  # High calorie targets
            min_cal = max(120, int(target_per_recipe * 0.8))
            max_cal = int(target_per_recipe * max_multiplier)
            category = "High Calorie"

        else:  # Very high calorie targets
            min_cal = max(150, int(target_per_recipe * 0.9))
            max_cal = int(target_per_recipe * max_multiplier)
            category = "Very High Calorie"

        print(f"Category: {category}, Range: {min_cal}-{max_cal} kcal")

        # Apply filtering
        calorie_mask = self.meal_data['nutrition'].apply(
            lambda x: min_cal <= x.get('calories', 0) <= max_cal
        )
        filtered_data = self.meal_data[calorie_mask]

        # SMART FALLBACK (Improved)
        if len(filtered_data) < 15:
            print(f"Only {len(filtered_data)} recipes, applying smart fallback...")

            if goal == 'lose':
                # For weight loss: expand downward more than upward
                fallback_min = max(30, int(min_cal * 0.5))
                fallback_max = int(max_cal * 1.2)
            elif goal == 'gain':
                # For weight gain: expand upward more than downward
                fallback_min = max(50, int(min_cal * 0.7))
                fallback_max = int(max_cal * 1.5)
            else:  # maintain
                # For maintenance: balanced expansion
                fallback_min = max(50, int(min_cal * 0.6))
                fallback_max = int(max_cal * 1.3)

            fallback_mask = self.meal_data['nutrition'].apply(
                lambda x: fallback_min <= x.get('calories', 0) <= fallback_max
            )
            filtered_data = self.meal_data[fallback_mask]
            print(f"Fallback range: {fallback_min}-{fallback_max} kcal, Found: {len(filtered_data)}")

        # Ultimate fallback: goal-aware recipe selection
        if len(filtered_data) < 10:
            print("Using goal-aware ultimate fallback...")
            calories_series = self.meal_data['nutrition'].apply(lambda x: x.get('calories', 0))

            if goal == 'lose':
                # Use bottom 60% for weight loss
                threshold = calories_series.quantile(0.6)
                mask = self.meal_data['nutrition'].apply(lambda x: x.get('calories', 0) <= threshold)
            elif goal == 'gain':
                # Use top 60% for weight gain
                threshold = calories_series.quantile(0.4)
                mask = self.meal_data['nutrition'].apply(lambda x: x.get('calories', 0) >= threshold)
            else:
                # Use middle 80% for maintenance
                low_threshold = calories_series.quantile(0.1)
                high_threshold = calories_series.quantile(0.9)
                mask = self.meal_data['nutrition'].apply(
                    lambda x: low_threshold <= x.get('calories', 0) <= high_threshold
                )

            filtered_data = self.meal_data[mask]
            print(f"Goal-aware fallback: {len(filtered_data)} recipes")

        # Final validation and stats
        if len(filtered_data) > 0:
            sample_calories = filtered_data['nutrition'].apply(lambda x: x.get('calories', 0))
            print(f"Final recipes: {len(filtered_data)}")
            print(f"Calorie range: {sample_calories.min():.0f}-{sample_calories.max():.0f} kcal")
            print(f"Average: {sample_calories.mean():.0f} kcal (target: {target_per_recipe:.0f})")

        self.filtered_meal_data = filtered_data
        self.meal_ids = filtered_data['ID'].unique().tolist()

        return len(filtered_data)

    def generate_meal_plan(self):
        """Generate an optimized meal plan"""
        best_schedule, nutrition, score = self.optimize()

        meal_plan = []

        for day_idx, day_meals in enumerate(best_schedule):
            day_plan = {
                "day": day_idx + 1,
                "meals": []
            }

            daily_nutrition = {
                'calories': 0,
                'protein': 0,
                'fat': 0,
                'carbohydrates': 0,
                'fiber': 0,
            }

            for meal_idx, meal_recipes in enumerate(day_meals):
                meal_info = {
                    "meal_number": meal_idx + 1,
                    "recipes": []
                }

                meal_nutrition = {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0,
                }

                for recipe_id in meal_recipes:
                    meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(recipe_id)]

                    if not meal_record.empty:
                        record = meal_record.iloc[0]

                        if 'Adjusted_Total_Nutrition' in record:
                            recipe_nutrition = record['Adjusted_Total_Nutrition']
                        else:
                            recipe_nutrition = record['nutrition']

                        recipe_info = {
                            "meal_id": recipe_id,
                            "title": record['Title'],
                            "nutrition": {
                                "calories": recipe_nutrition.get('calories', 0),
                                "protein": recipe_nutrition.get('protein', 0),
                                "fat": recipe_nutrition.get('fat', 0),
                                "carbohydrates": recipe_nutrition.get('carbohydrates', 0),
                                "fiber": recipe_nutrition.get('fiber', 0),
                            }
                        }

                        meal_nutrition['calories'] += recipe_nutrition.get('calories', 0)
                        meal_nutrition['protein'] += recipe_nutrition.get('protein', 0)
                        meal_nutrition['fat'] += recipe_nutrition.get('fat', 0)
                        meal_nutrition['carbohydrates'] += recipe_nutrition.get('carbohydrates', 0)
                        meal_nutrition['fiber'] += recipe_nutrition.get('fiber', 0)

                        meal_info["recipes"].append(recipe_info)

                meal_info["meal_nutrition"] = meal_nutrition

                daily_nutrition['calories'] += meal_nutrition['calories']
                daily_nutrition['protein'] += meal_nutrition['protein']
                daily_nutrition['fat'] += meal_nutrition['fat']
                daily_nutrition['carbohydrates'] += meal_nutrition['carbohydrates']
                daily_nutrition['fiber'] += meal_nutrition['fiber']

                day_plan["meals"].append(meal_info)

            day_plan["daily_nutrition"] = daily_nutrition
            meal_plan.append(day_plan)

        result = {
            "meal_plan": meal_plan,
            "average_daily_nutrition": nutrition,
            "target_nutrition": self.target_metrics,
            "fitness_score": score
        }

        return result


class MealScheduler:
    def __init__(self, model_dir='models/'):
        """Initialize the meal scheduler"""
        self.model_dir = model_dir
        self.cbf_recommender = CBFRecommender(model_dir)

    def create_user_preferences(self, excluded_ingredients=None, dietary_type=None,
                                min_nutrition=None, max_nutrition=None):
        """Create user preferences object for filtering recipes"""
        return UserPreferences(
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type,
            min_nutrition=min_nutrition,
            max_nutrition=max_nutrition
        )

    def generate_meal_plan(self, age, gender, weight, height, activity_level,
                           meals_per_day=3, recipes_per_meal=3, goal='maintain', excluded_ingredients=None,
                           dietary_type=None, min_nutrition=None, max_nutrition=None):
        """Generate optimized meal plan dengan MLflow tracking fallback"""

        with start_run(run_name="PSO Meal Planning", experiment_name="Meal Planning"):
            try:
                user_preferences = self.create_user_preferences(
                    excluded_ingredients=excluded_ingredients,
                    dietary_type=dietary_type,
                    min_nutrition=min_nutrition,
                    max_nutrition=max_nutrition
                )

                self.pso_optimizer = ParticleSwarmOptimizer(
                    num_particles=30,
                    num_days=7,
                    meals_per_day=meals_per_day,
                    recipes_per_meal=recipes_per_meal,
                    max_iterations=50
                )

                self.pso_optimizer.set_user_requirements(
                    age=age,
                    gender=gender,
                    weight=weight,
                    height=height,
                    activity_level=activity_level,
                    goal=goal,
                    user_preferences=user_preferences
                )

                self.filtered_meal_data = self.pso_optimizer.filtered_meal_data
                self.target_metrics = self.pso_optimizer.target_metrics

                best_schedule, nutrition, score = self.pso_optimizer.optimize()

                # Log basic metrics
                basic_metrics = {
                    "final_calories": nutrition['calories'],
                    "final_protein": nutrition['protein'],
                    "fitness_score": score
                }
                log_metrics(basic_metrics)

                # Format meal plan
                meal_plan = []

                for day_idx, day_meals in enumerate(best_schedule):
                    day_plan = {
                        "day": day_idx + 1,
                        "meals": []
                    }

                    daily_nutrition = {
                        'calories': 0,
                        'protein': 0,
                        'fat': 0,
                        'carbohydrates': 0,
                        'fiber': 0,
                    }

                    for meal_idx, meal_recipes in enumerate(day_meals):
                        meal_info = {
                            "meal_number": meal_idx + 1,
                            "recipes": []
                        }

                        meal_nutrition = {
                            'calories': 0,
                            'protein': 0,
                            'fat': 0,
                            'carbohydrates': 0,
                            'fiber': 0,
                        }

                        for recipe_id in meal_recipes:
                            meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(recipe_id)]

                            if not meal_record.empty:
                                recipe_info = {
                                    "meal_id": recipe_id,
                                    "title": meal_record.iloc[0]['Title'],
                                    "nutrition": meal_record.iloc[0]['nutrition']
                                }

                                nutrition = meal_record.iloc[0]['nutrition']
                                meal_nutrition['calories'] += nutrition.get('calories', 0)
                                meal_nutrition['protein'] += nutrition.get('protein', 0)
                                meal_nutrition['fat'] += nutrition.get('fat', 0)
                                meal_nutrition['carbohydrates'] += nutrition.get('carbohydrates', 0)
                                meal_nutrition['fiber'] += nutrition.get('fiber', 0)

                                meal_info["recipes"].append(recipe_info)

                        meal_info["meal_nutrition"] = meal_nutrition

                        daily_nutrition['calories'] += meal_nutrition['calories']
                        daily_nutrition['protein'] += meal_nutrition['protein']
                        daily_nutrition['fat'] += meal_nutrition['fat']
                        daily_nutrition['carbohydrates'] += meal_nutrition['carbohydrates']
                        daily_nutrition['fiber'] += meal_nutrition['fiber']

                        day_plan["meals"].append(meal_info)

                    day_plan["daily_nutrition"] = daily_nutrition
                    meal_plan.append(day_plan)

                result = {
                    "meal_plan": meal_plan,
                    "average_daily_nutrition": nutrition,
                    "target_nutrition": self.target_metrics,
                    "fitness_score": score
                }

                # Save meal plan sebagai artifact
                try:
                    temp_path = os.path.abspath("temp_meal_plan.json")
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

                    with open(temp_path, 'w') as f:
                        json.dump(result, f, indent=2)

                    log_artifact(temp_path, "optimization_results")

                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not save meal plan artifact: {e}")

                set_tag("status", "success")
                return result

            except Exception as e:
                set_tag("status", "failed")
                log_params({"error": str(e)})
                raise

    def save_meal_plan(self, meal_plan, output_file='output/meal_plan.json'):
        """Save meal plan to a JSON file dengan MLflow logging"""
        output_file = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert numpy values to Python types for JSON serialization
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj

        meal_plan_serializable = convert_numpy_to_python(meal_plan)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(meal_plan_serializable, f, indent=2, ensure_ascii=False)

        # Log to MLflow jika tersedia
        try:
            log_artifact(output_file, "meal_plans")
        except Exception as e:
            print(f"Warning: Could not log meal plan to MLflow: {e}")

        print(f"Meal plan saved to {output_file}")