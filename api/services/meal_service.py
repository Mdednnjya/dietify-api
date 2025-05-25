import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.pso_meal_scheduler import MealScheduler
from api.models.user_models import UserProfile

try:
    from src.utils.mlflow_manager import start_run, log_params, log_metrics, set_tag

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


    def start_run(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()


    def log_params(*args, **kwargs):
        pass


    def log_metrics(*args, **kwargs):
        pass


    def set_tag(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)


class MealPlanService:
    def __init__(self):
        self.scheduler = MealScheduler()

    async def process_meal_plan_async(
            self,
            session_id: str,
            user_profile: UserProfile,
            redis_client
    ):
        """Background task untuk generate meal plan dengan MLflow fallback"""

        # MLflow tracking dengan fallback mode
        with start_run(run_name=f"API Request {session_id}", experiment_name="Meal Planning API"):
            try:
                # Log request start
                request_params = {
                    "session_id": session_id,
                    "user_age": user_profile.age,
                    "user_gender": user_profile.gender.value,
                    "user_weight": user_profile.weight,
                    "user_height": user_profile.height,
                    "activity_level": user_profile.activity_level.value,
                    "goal": user_profile.goal.value,
                    "meals_per_day": user_profile.meals_per_day,
                    "recipes_per_meal": user_profile.recipes_per_meal,
                    "excluded_ingredients": ",".join(user_profile.exclude) if user_profile.exclude else "none",
                    "diet_type": user_profile.diet_type.value if user_profile.diet_type else "none"
                }
                log_params(request_params)

                # Set processing start time
                await redis_client.setex(f"started:{session_id}", 3600, "true")

                # Log start metrics
                log_metrics({"request_started": 1})
                set_tag("session_id", session_id)
                set_tag("status", "processing")

                # Run PSO optimization dengan timeout
                meal_plan = await asyncio.wait_for(
                    self._generate_meal_plan(user_profile),
                    timeout=60.0  # 1 minute timeout
                )

                # Log successful completion
                log_metrics({
                    "optimization_success": 1,
                    "final_calories": meal_plan.get('average_daily_nutrition', {}).get('calories', 0),
                    "final_protein": meal_plan.get('average_daily_nutrition', {}).get('protein', 0),
                    "fitness_score": meal_plan.get('fitness_score', 0)
                })
                set_tag("status", "completed")

                # Cache result
                await redis_client.setex(
                    f"meal_plan:{session_id}",
                    3600,
                    json.dumps(meal_plan, ensure_ascii=False)
                )

                # Update status
                await redis_client.setex(f"status:{session_id}", 3600, "completed")

                logger.info(f"✅ Meal plan completed for session: {session_id}")

            except asyncio.TimeoutError:
                # Log timeout dan gunakan fallback
                log_metrics({"timeout_occurred": 1})
                set_tag("status", "timeout_fallback")

                fallback_plan = self._get_fallback_meal_plan(user_profile)
                await redis_client.setex(
                    f"meal_plan:{session_id}",
                    3600,
                    json.dumps(fallback_plan, ensure_ascii=False)
                )
                await redis_client.setex(f"status:{session_id}", 3600, "completed")
                logger.warning(f"⚠️ Timeout - using fallback for session: {session_id}")

            except Exception as e:
                # Log error dan handle gracefully
                error_msg = f"Generation failed: {str(e)}"

                log_metrics({"error_occurred": 1})
                log_params({"error_message": error_msg})
                set_tag("status", "error")

                await redis_client.setex(f"error:{session_id}", 3600, error_msg)
                await redis_client.setex(f"status:{session_id}", 3600, "error")
                logger.error(f"❌ Error for session {session_id}: {error_msg}")

    async def _generate_meal_plan(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate meal plan using existing PSO scheduler"""
        loop = asyncio.get_event_loop()

        # Run PSO optimization di thread pool
        meal_plan = await loop.run_in_executor(
            None,
            self.scheduler.generate_meal_plan,
            user_profile.age,
            user_profile.gender.value,
            user_profile.weight,
            user_profile.height,
            user_profile.activity_level.value,
            user_profile.meals_per_day,
            user_profile.recipes_per_meal,
            user_profile.goal.value,
            user_profile.exclude,
            user_profile.diet_type.value if user_profile.diet_type else None
        )

        # FIX: Recalculate average daily nutrition correctly
        if 'meal_plan' in meal_plan:
            daily_totals = []
            for day in meal_plan['meal_plan']:
                daily_totals.append(day['daily_nutrition'])

            # Calculate correct average
            avg_nutrition = {
                'calories': sum(day['calories'] for day in daily_totals) / len(daily_totals),
                'protein': sum(day['protein'] for day in daily_totals) / len(daily_totals),
                'fat': sum(day['fat'] for day in daily_totals) / len(daily_totals),
                'carbohydrates': sum(day['carbohydrates'] for day in daily_totals) / len(daily_totals),
                'fiber': sum(day['fiber'] for day in daily_totals) / len(daily_totals)
            }

            # Update with correct average
            meal_plan['average_daily_nutrition'] = avg_nutrition

        return meal_plan

    def _get_fallback_meal_plan(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Fallback meal plan untuk demo dengan user profile"""

        # Simple nutritional calculation untuk fallback
        base_calories = 1800 if user_profile.gender.value == 'female' else 2200

        if user_profile.goal.value == 'lose':
            target_calories = base_calories * 0.85
        elif user_profile.goal.value == 'gain':
            target_calories = base_calories * 1.15
        else:
            target_calories = base_calories

        # Fallback meal plan dengan proper structure
        fallback_meals = []

        for day in range(1, 8):  # 7 days
            day_plan = {
                "day": day,
                "meals": []
            }

            daily_nutrition = {
                'calories': 0,
                'protein': 0,
                'fat': 0,
                'carbohydrates': 0,
                'fiber': 0
            }

            # Generate meals for each day
            for meal_num in range(1, user_profile.meals_per_day + 1):
                meal_calories = target_calories / user_profile.meals_per_day

                meal_info = {
                    "meal_number": meal_num,
                    "recipes": []
                }

                meal_nutrition = {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0
                }

                # Generate recipes for each meal
                for recipe_num in range(user_profile.recipes_per_meal):
                    recipe_calories = meal_calories / user_profile.recipes_per_meal

                    recipe_info = {
                        "meal_id": f"fallback_{day}_{meal_num}_{recipe_num}",
                        "title": f"Healthy Sample Recipe {recipe_num + 1}",
                        "nutrition": {
                            "calories": recipe_calories,
                            "protein": recipe_calories * 0.2 / 4,
                            "fat": recipe_calories * 0.25 / 9,
                            "carbohydrates": recipe_calories * 0.55 / 4,
                            "fiber": min(10, recipe_calories / 100)
                        }
                    }

                    meal_info["recipes"].append(recipe_info)

                    # Add to meal nutrition
                    for nutrient in meal_nutrition:
                        meal_nutrition[nutrient] += recipe_info["nutrition"][nutrient]

                meal_info["meal_nutrition"] = meal_nutrition
                day_plan["meals"].append(meal_info)

                # Add to daily nutrition
                for nutrient in daily_nutrition:
                    daily_nutrition[nutrient] += meal_nutrition[nutrient]

            day_plan["daily_nutrition"] = daily_nutrition
            fallback_meals.append(day_plan)

        # Calculate average daily nutrition
        avg_nutrition = {
            'calories': sum(day['daily_nutrition']['calories'] for day in fallback_meals) / 7,
            'protein': sum(day['daily_nutrition']['protein'] for day in fallback_meals) / 7,
            'fat': sum(day['daily_nutrition']['fat'] for day in fallback_meals) / 7,
            'carbohydrates': sum(day['daily_nutrition']['carbohydrates'] for day in fallback_meals) / 7,
            'fiber': sum(day['daily_nutrition']['fiber'] for day in fallback_meals) / 7
        }

        return {
            "status": "fallback",
            "message": "Using optimized sample meal plan due to timeout",
            "meal_plan": fallback_meals,
            "average_daily_nutrition": avg_nutrition,
            "target_nutrition": {
                "calories": target_calories,
                "protein": target_calories * 0.2 / 4,
                "fat": target_calories * 0.25 / 9,
                "carbohydrates": target_calories * 0.55 / 4,
                "fiber": 25
            },
            "fitness_score": 0.85,  # Reasonable fallback score
            "user_profile": user_profile.dict()
        }