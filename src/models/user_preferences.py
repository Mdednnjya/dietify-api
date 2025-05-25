class UserPreferences:
    def __init__(self,
                 excluded_ingredients=None,
                 dietary_type=None,
                 min_nutrition=None,
                 max_nutrition=None):
        """
        Initialize user preferences for food recommendations

        Args:
            excluded_ingredients: List of ingredients to exclude (e.g., ["chicken", "beef"])
            dietary_type: Type of diet (e.g., "vegetarian", "vegan", "pescatarian")
            min_nutrition: Dict with minimum required nutrition values
            max_nutrition: Dict with maximum allowed nutrition values
        """
        self.excluded_ingredients = excluded_ingredients or []
        self.dietary_type = dietary_type
        self.min_nutrition = min_nutrition or {}
        self.max_nutrition = max_nutrition or {}

        # Convert all excluded ingredients to lowercase for matching
        self.excluded_ingredients = [ing.lower() for ing in self.excluded_ingredients]

        # Define ingredient exclusions based on dietary types
        self.dietary_exclusions = {
            # Vegetarian: No meat, poultry, or seafood (plants, dairy, eggs OK)
            "vegetarian": [
                # Daging & Unggas
                "chicken", "beef", "pork", "daging", "lamb", "kambing", "ayam", "sapi",
                "tulang ayam", "nugget", "geprek", "rica", "betutu", "katsu",
                "tulang ayam rica wangi", "ayam geprek", "ayam woku manado"

                # Seafood (ikan & udang)
                "ikan", "fish", "udang", "shrimp", "patin", "gurame", "teri", "tuna",
                "bandeng", "tenggiri", "siomay",

                # Processed meat
                "kornet", "abon", "sate", "rawon", "tongseng", "bulgogi"
            ],

            # Vegan: No animal products at all (only plants)
            "vegan": [
                # All from vegetarian PLUS dairy & eggs
                "chicken", "beef", "pork", "daging", "lamb", "kambing", "ayam", "sapi",
                "tulang ayam", "nugget", "geprek", "rica", "betutu", "katsu",
                "ikan", "fish", "udang", "shrimp", "patin", "gurame", "teri", "tuna",
                "bandeng", "tenggiri", "siomay", "kornet", "abon", "sate", "rawon",
                "tongseng", "bulgogi", "tulang ayam rica wangi", "ayam geprek",
                "ayam woku manado"

                # Dairy & Eggs
                "egg", "telur", "telor", "dadar", "milk", "susu", "cheese", "keju",
                "yogurt", "yoghurt", "opor"
            ],

            # Pescatarian: No meat or poultry, but seafood OK (fish, dairy, eggs OK)
            "pescatarian": [
                # Only land animals (keep seafood)
                "chicken", "beef", "pork", "daging", "lamb", "kambing", "ayam", "sapi",
                "tulang ayam", "nugget", "geprek", "rica", "betutu", "katsu",
                "kornet", "abon", "sate", "rawon", "tongseng", "bulgogi",
                "tulang ayam rica wangi", "ayam geprek", "ayam woku manado"
                # NO ikan/udang - pescatarian can eat seafood
            ]
        }

        # Add dietary-type based exclusions
        if self.dietary_type and self.dietary_type.lower() in self.dietary_exclusions:
            self.excluded_ingredients.extend(self.dietary_exclusions[self.dietary_type.lower()])

        # Remove duplicates while preserving order
        self.excluded_ingredients = list(dict.fromkeys(self.excluded_ingredients))

    def filter_recipes(self, recipe_df):
        """
        Filter recipes based on user preferences

        Args:
            recipe_df: DataFrame containing recipes with "Ingredients_Enriched" column

        Returns:
            DataFrame with filtered recipes
        """
        filtered_df = recipe_df.copy()

        # Filter by excluded ingredients if provided
        if self.excluded_ingredients:
            # Create mask for recipes containing excluded ingredients
            if 'Ingredients_Enriched' in filtered_df.columns:
                # For enriched format
                contains_excluded = filtered_df['Ingredients_Enriched'].apply(
                    lambda ingredients: any(
                        any(excluded in ing.get('ingredient', '').lower() for excluded in self.excluded_ingredients)
                        for ing in ingredients
                    )
                )
            elif 'Ingredients_Parsed' in filtered_df.columns:
                # For parsed format
                contains_excluded = filtered_df['Ingredients_Parsed'].apply(
                    lambda ingredients: any(
                        any(excluded in ing.lower() for excluded in self.excluded_ingredients)
                        for ing in ingredients
                    )
                )
            elif 'ingredient_text' in filtered_df.columns:
                # For text format
                contains_excluded = filtered_df['ingredient_text'].apply(
                    lambda text: any(
                        excluded in text.lower() for excluded in self.excluded_ingredients
                    )
                )
            else:
                # For format with just nutrition data
                contains_excluded = filtered_df['Title'].apply(
                    lambda title: any(
                        excluded in title.lower() for excluded in self.excluded_ingredients
                    )
                )

            # Filter out recipes containing excluded ingredients
            filtered_df = filtered_df[~contains_excluded]

        # Filter by nutrition constraints
        if 'nutrition' in filtered_df.columns:
            # For min constraints
            for nutrient, min_value in self.min_nutrition.items():
                if min_value is not None:
                    filtered_df = filtered_df[
                        filtered_df['nutrition'].apply(
                            lambda x: x.get(nutrient, 0) >= min_value
                        )
                    ]

            # For max constraints
            for nutrient, max_value in self.max_nutrition.items():
                if max_value is not None:
                    filtered_df = filtered_df[
                        filtered_df['nutrition'].apply(
                            lambda x: x.get(nutrient, 0) <= max_value
                        )
                    ]

        return filtered_df