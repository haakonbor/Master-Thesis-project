import bpy
from numpy import random
from enum import IntEnum
import copy
import json
import os

""" --------------------------------------------------------- """
""" ---------------------- ENCODING ------------------------- """
""" --------------------------------------------------------- """
class Features(IntEnum):
    SEED = 0
    EXTRUSIONS = 1
    MIRROR_X = 2
    MIRROR_Y = 3
    MIRROR_Z = 4
    SUBDIVIDE = 5
    BEVEL = 6
    EXTRUDE_MIN = 7
    EXTRUDE_MAX = 8
    TAPER_MIN = 9
    TAPER_MAX = 10
    ROTATION_MIN = 11
    ROTATION_MAX = 12
    FAVOUR_X = 13
    FAVOUR_Y = 14
    FAVOUR_Z = 15
    SCALING_X = 16
    SCALING_Y = 17
    SCALING_Z = 18
    
n_integer_features = 7

boundaries = [None] * len(Features)
boundaries[Features.SEED] = { "min": 0, "max": 2000 }
boundaries[Features.EXTRUSIONS] = { "min": 5, "max": 35 }
boundaries[Features.MIRROR_X] = { "min": 0, "max": 2 }
boundaries[Features.MIRROR_Y] = { "min": 0, "max": 2 }
boundaries[Features.MIRROR_Z] = { "min": 0, "max": 2 }
boundaries[Features.SUBDIVIDE] = { "min": 0, "max": 2 }
boundaries[Features.BEVEL] = { "min": 0, "max": 2 }
boundaries[Features.EXTRUDE_MIN] = { "min": 0.3, "max": 1.0 }
boundaries[Features.EXTRUDE_MAX] = { "min": 0.3, "max": 1.0 }
boundaries[Features.TAPER_MIN] = { "min": 0.3, "max": 1.0 }
boundaries[Features.TAPER_MAX] = { "min": 0.3, "max": 1.0 }
boundaries[Features.ROTATION_MIN] = { "min": -45.0, "max": 45.0 }
boundaries[Features.ROTATION_MAX] = { "min": -45.0, "max": 45.0 }
boundaries[Features.FAVOUR_X] = { "min": 0.0, "max": 1.0 }
boundaries[Features.FAVOUR_Y] = { "min": 0.0, "max": 1.0 }
boundaries[Features.FAVOUR_Z] = { "min": 0.0, "max": 1.0 }
boundaries[Features.SCALING_X] = { "min": 0.5, "max": 1.0 }
boundaries[Features.SCALING_Y] = { "min": 0.5, "max": 1.0 }
boundaries[Features.SCALING_Z] = { "min": 0.5, "max": 1.0 }

min_boundaries = [Features.EXTRUDE_MIN, Features.TAPER_MIN, Features.ROTATION_MIN]
max_boundaries = [Features.EXTRUDE_MAX, Features.TAPER_MAX, Features.ROTATION_MAX]


""" --------------------------------------------------------- """
""" ---------------------- SYSTEM PARAMETERS ---------------- """
""" --------------------------------------------------------- """
grid_size = 4
spacing = 6

random.seed(0)

population_size = 200
crossover_rate = 0.9
mutation_rate = 1 / len(Features)

current_population = []
rated_shapes = []

default_filename = "rated_shapes.json"

k = 15
weighted_knn = True

current_generation = 1


""" --------------------------------------------------------- """
""" ---------------------- SHAPE CLASS ---------------------- """
""" --------------------------------------------------------- """
class Shape:
    def __init__(self, liked, seed, extrusions, mirror, subdivide, bevel ,extrude_min,
                 extrude_max, taper_min, taper_max, rotation_min, rotation_max, favour, scaling): 
        self.features = [
            seed,
            extrusions,
            mirror[0],
            mirror[1],
            mirror[2],
            subdivide, 
            bevel,
            extrude_min,
            extrude_max,
            taper_min, 
            taper_max, 
            rotation_min, 
            rotation_max,
            favour[0],
            favour[1],
            favour[2],
            scaling[0],
            scaling[1],
            scaling[2],
        ]
        
        self.liked = liked
        self.fitness = 100 if liked == 1 else 1 if liked == -1 else None
    
    @classmethod
    def from_dict(cls, shape_dict):
        liked = shape_dict['liked']
        features = shape_dict.get('features', None)
        seed = features[Features.SEED]
        extrusions = features[Features.EXTRUSIONS]
        mirror = features[Features.MIRROR_X : Features.MIRROR_Z + 1]
        subdivide = features[Features.SUBDIVIDE]
        bevel = features[Features.BEVEL]
        extrude_min = features[Features.EXTRUDE_MIN]
        extrude_max = features[Features.EXTRUDE_MAX]
        taper_min = features[Features.TAPER_MIN]
        taper_max = features[Features.TAPER_MAX]
        rotation_min = features[Features.ROTATION_MIN]
        rotation_max = features[Features.ROTATION_MAX]
        favour = features[Features.FAVOUR_X : Features.FAVOUR_Z + 1]
        scaling = features[Features.SCALING_X : Features.SCALING_Z + 1]
        return cls(liked, seed, extrusions, mirror, subdivide, bevel, extrude_min, extrude_max, taper_min, taper_max, rotation_min, rotation_max, favour, scaling)
    
    def evaluate_fitness(self):
        global rated_shapes
        
        if self.fitness != None:
            return self.fitness
        
        elif len(rated_shapes) > k:
            # Get K nearest neighbours of shape
            rated_shapes.sort(key=self.get_distance_to_shape)
            knn = rated_shapes[:k]
            
            # Get sum of each classification
            liked_sum = 0
            disliked_sum = 0
            
            # Distance weighted K-NN  
            if weighted_knn:
                epsilon = 1e-6
                for neighbour in knn:
                    distance = self.get_distance_to_shape(neighbour)
                    if neighbour.liked == 1:
                        liked_sum += (1 / (distance + epsilon))
                    elif neighbour.liked == -1:
                        disliked_sum += (1 / (distance + epsilon))
                    else:
                        print("UNRATED SHAPE FOUND IN RATED SHAPES LIST!") 
            
            # Standard K-NN
            else:
                for neighbour in knn:
                    if neighbour.liked == 1:
                        liked_sum += 1
                    elif neighbour.liked == -1:
                        disliked_sum += 1
                    else:
                        print("UNRATED SHAPE FOUND IN RATED SHAPES LIST!") 
            
            # Predict fitness from K-NN classification
            predicted_fitness = 100 * liked_sum / (liked_sum + disliked_sum) + 1 * disliked_sum / (liked_sum + disliked_sum)
            self.fitness = predicted_fitness
            return predicted_fitness
        else:
            #relative_size = sum(self.features[-3:]) / 3
            #return 100 * relative_size
            return 50
        
    def get_distance_to_shape(self, shape):
        distance = 0
        
        for i in range(len(self.features)):
            distance += ((self.features[i] - shape.features[i]) / (boundaries[i]["max"] - boundaries[i]["min"])) ** 2
            
        return distance
    

""" --------------------------------------------------------------- """
""" ---------------------- GENETIC ALGORITHM ---------------------- """
""" --------------------------------------------------------------- """
def roulette_select_parent(shapes):
    max = sum([shape.evaluate_fitness() for shape in shapes])
    selection_probs = [shape.evaluate_fitness()/max for shape in shapes]
    parent_index = random.choice(len(shapes), p=selection_probs)
    return shapes[parent_index]

def crossover(parent1, parent2):
    child1 = Shape()
    child2 = Shape()
    
    if random.random() < crossover_rate:
        cross_point = random.randint(1, len(parent1.features) - 2)
        child1.features = parent1.features[:cross_point] + parent2.features[cross_point:]
        child2.features = parent2.features[:cross_point] + parent1.features[cross_point:]
        
    return [child1, child2]

def multi_crossover(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child1.liked = 0
    child1.fitness = None
    child2 = copy.deepcopy(parent2)
    child2.liked = 0
    child2.fitness = None
    
    if random.random() < crossover_rate:
        cross_point1 = random.randint(1, len(parent1.features) - 3)
        cross_point2 = random.randint(cross_point1, len(parent1.features) - 2)
        child1.features = parent1.features[:cross_point1] + parent2.features[cross_point1:cross_point2] + parent1.features[cross_point2:]
        child2.features = parent2.features[:cross_point1] + parent1.features[cross_point1:cross_point2] + parent2.features[cross_point2:]
        
        # Check if boundary rules are broken
        for i in min_boundaries:
            if child1.features[i] > child1.features[i+1]:
                child1.features[i+1] = child1.features[i]
                
            if child2.features[i] > child2.features[i+1]:
                child2.features[i+1] = child2.features[i]
            
        
    return [child1, child2]


def mutation(shape):
    for i in range(len(shape.features)):
        if random.random() < mutation_rate:
            if i in max_boundaries:
                feature_val = get_random_feature_value(i, min=shape.features[i-1])
            elif i in min_boundaries:
                feature_val = get_random_feature_value(i, max=shape.features[i+1])
            else:
                feature_val = get_random_feature_value(i)
                
            shape.features[i] = feature_val
            

def get_random_feature_value(i, min=None, max=None):
    random_val = None
    
    if i < n_integer_features:
        random_val = random.randint(boundaries[i]["min"], boundaries[i]["max"])
    elif min != None:
        random_val = round(random.uniform(min, boundaries[i]["max"]), 3)
    elif max != None:
        random_val = round(random.uniform(boundaries[i]["min"], max), 3)
    else:
        random_val = round(random.uniform(boundaries[i]["min"], boundaries[i]["max"]), 3)

    return random_val

   
""" --------------------------------------------------- """
""" ---------------------- SCENE ---------------------- """
""" --------------------------------------------------- """ 

def convergence(treshold = 50):
    top_16_avg_fitness = average_fitness(current_population[:grid_size*grid_size])
    print("TOP 16 AVERAGE FITNESS:", top_16_avg_fitness)
    if top_16_avg_fitness > treshold:
        return True
    else:
        return False


def run_generations(treshold = 50, generations = 100):
    for i in range(0, generations):
        generate_new_set(create=False)
        if convergence(treshold):
            insert_shapes_into_scene()
            return
    
    generate_new_set()
       
        
def average_fitness(population):
    fitness_values = [shape.evaluate_fitness() for shape in population]
    return sum(fitness_values) / len(fitness_values)

   
def generate_new_set(create=True):
    global current_population
    global current_generation
    
    print("GENERATION:", current_generation)
    current_generation += 1
    
    # Get rated shapes from Blender scene
    for obj in bpy.context.scene.objects:
         if obj.name.startswith("Shape_") and obj["liked"] != 0:
            rated_shape_index = int(obj.name.split("_")[1]) - 1
            current_population[rated_shape_index].liked = obj["liked"]
            rated_shapes.append(current_population[rated_shape_index])
            
    clear_scene()
    new_population = []
    
    generation_fitness = 0
    for i in range(0, population_size, 2):
        parent1 = roulette_select_parent(current_population)
        parent2 = roulette_select_parent(current_population)
        for child in multi_crossover(parent1, parent2):
            mutation(child)
            generation_fitness += child.evaluate_fitness()
            new_population.append(child)
            
    print("TOTAL AVERAGE FITNESS:", generation_fitness / population_size)
                
    current_population = new_population
    current_population.sort(reverse=True, key=lambda s: s.evaluate_fitness())
    
    if create:
        insert_shapes_into_scene()


def insert_shapes_into_scene():
    shape_count = 0
    for i in range(grid_size):
            for j in range(grid_size):
                shape = current_population[shape_count]
                shape_count += 1
                create_shape( j * spacing, -i * spacing, i * grid_size + j, shape)        

def create_label(x, z, index):
    label_name = f"Label_{index}"
    label_size = spacing * 0.1

    # Check if the label already exists
    if label_name in bpy.data.objects:
        label = bpy.data.objects[label_name]
        label.location = (x - label_size * 2 - 1.5*spacing, -2, z - 2 + 1.5*spacing)
    else:
        bpy.ops.object.text_add(location=(x - label_size * 2 - 1.5*spacing, -2, z - 2 + 1.5*spacing))
        label = bpy.context.active_object
        label.rotation_euler = (1.5708, 0, 0)  # Rotate 90 degrees along the X-axis
        label.name = label_name

    label.data.body = f"Shape {index + 1}"
    label.data.size = label_size


def create_shape(x, z, index, shape_instance):
    shape_name = f"Shape_{index + 1}"
    
    if shape_name in bpy.data.objects:
        shape = bpy.data.objects[shape_name]
    else:
        shape = bpy.context.active_object
        bpy.ops.mesh.shape_generator()
        
    collection = bpy.data.collections.get("Generated Shape Collection")
    
    # Collection properties
    collection.name = f"Shape_{index + 1}_collection"
    collection.shape_generator_properties.random_seed = shape_instance.features[Features.SEED]
    collection.shape_generator_properties.amount = 0    # trick to speed up shape generation
    collection.shape_generator_properties.mirror_x = shape_instance.features[Features.MIRROR_X] == 1
    collection.shape_generator_properties.mirror_y = shape_instance.features[Features.MIRROR_Y] == 1
    collection.shape_generator_properties.mirror_z = shape_instance.features[Features.MIRROR_Z] == 1
    collection.shape_generator_properties.subsurf_subdivisions = 1
    collection.shape_generator_properties.is_subsurf = shape_instance.features[Features.SUBDIVIDE] == 1
    collection.shape_generator_properties.is_bevel = shape_instance.features[Features.BEVEL] == 1
    collection.shape_generator_properties.min_extrude = shape_instance.features[Features.EXTRUDE_MIN]
    collection.shape_generator_properties.max_extrude = shape_instance.features[Features.EXTRUDE_MAX]
    collection.shape_generator_properties.min_taper = shape_instance.features[Features.TAPER_MIN]
    collection.shape_generator_properties.max_taper = shape_instance.features[Features.TAPER_MAX]
    collection.shape_generator_properties.min_rotation = shape_instance.features[Features.ROTATION_MIN]
    collection.shape_generator_properties.max_rotation = shape_instance.features[Features.ROTATION_MAX]
    collection.shape_generator_properties.favour_vec = [shape_instance.features[Features.FAVOUR_X], shape_instance.features[Features.FAVOUR_Y], shape_instance.features[Features.FAVOUR_Z]]
    collection.shape_generator_properties.scale = [shape_instance.features[Features.SCALING_X], shape_instance.features[Features.SCALING_Y], shape_instance.features[Features.SCALING_Z]]
    collection.shape_generator_properties.amount = shape_instance.features[Features.EXTRUSIONS] # trick to speed up shape generation
        
    collection.shape_generator_properties.show_seed_panel = True
    collection.shape_generator_properties.show_extrude_panel = True
    collection.shape_generator_properties.show_mirror_panel = True
    collection.shape_generator_properties.show_translation_panel = True
    
    # Shape properties
    shape = bpy.context.active_object
    shape.name = shape_name
    shape.location = (x - 1.5*spacing, 0, z + 1.5*spacing)
    shape["liked"] = 0
    
    # Shape material
    shape.data.materials.append(bpy.data.materials.new(name=f"Color_{index + 1}"))
    shape.data.materials[-1].diffuse_color = (random.random(), random.random(), random.random(), 1.0)
    shape.active_material_index = len(shape.data.materials) - 1
    
    print("SHAPE CREATED:", [round(feature, 1) for feature in shape_instance.features])
    print("EXPECTED FITNESS:", shape_instance.evaluate_fitness())
                                

def init_scene():
    global rated_shapes
    global default_filename    
    
    for i in range(population_size):
        random_seed = get_random_feature_value(Features.SEED)
        random_extrusions = get_random_feature_value(Features.EXTRUSIONS)
        random_mirror = [get_random_feature_value(Features.MIRROR_X), get_random_feature_value(Features.MIRROR_Y), get_random_feature_value(Features.MIRROR_Z)]
        random_subdivide = get_random_feature_value(Features.SUBDIVIDE)
        random_bevel = get_random_feature_value(Features.BEVEL)
        
        random_min_extrude = get_random_feature_value(Features.EXTRUDE_MIN)
        random_max_extrude = get_random_feature_value(Features.EXTRUDE_MAX, min=random_min_extrude)
        random_min_taper = get_random_feature_value(Features.TAPER_MIN)
        random_max_taper = get_random_feature_value(Features.TAPER_MAX, min=random_min_taper)
        random_min_rotation = get_random_feature_value(Features.ROTATION_MIN)
        random_max_rotation = get_random_feature_value(Features.ROTATION_MAX, min=random_min_rotation)
        random_favour = [get_random_feature_value(Features.FAVOUR_X), get_random_feature_value(Features.FAVOUR_Y), get_random_feature_value(Features.FAVOUR_Z)] 
        random_scaling = [get_random_feature_value(Features.SCALING_X), get_random_feature_value(Features.SCALING_Y), get_random_feature_value(Features.SCALING_Z)] 
        
        random_shape = Shape(
            liked=0,
            seed=random_seed,
            extrusions=random_extrusions,
            mirror=random_mirror,
            subdivide=random_subdivide, 
            bevel=random_bevel, 
            extrude_min=random_min_extrude,
            extrude_max=random_max_extrude,
            taper_min=random_min_taper,
            taper_max=random_max_taper,
            rotation_min=random_min_rotation,
            rotation_max=random_max_rotation,
            favour=random_favour,
            scaling=random_scaling
        )
        
        current_population.append(random_shape)
    
    shape_count = 0
    current_population.sort(reverse=True, key=lambda s: s.evaluate_fitness())
    for i in range(grid_size):
        for j in range(grid_size):
            current_shape = current_population[shape_count]
            create_shape(j * spacing, -i * spacing, i * grid_size + j, current_shape)
            create_label(j * spacing, -i * spacing, i * grid_size + j)
            shape_count += 1


def save_shapes(filename):
    global rated_shapes
    
    blend_file_directory = os.path.dirname(bpy.data.filepath)  # Get the directory of the Blender file
    file_path = os.path.join(blend_file_directory, filename)
    
    with open(file_path, "w") as file:
        json.dump([shape.__dict__ for shape in rated_shapes], file)
    
    file.close()
    
    print(f"FILE WRITTEN: '{file_path}' with {len(rated_shapes)} ratings")
    
    
def load_shapes(filename):
    global rated_shapes

    blend_file_directory = os.path.dirname(bpy.data.filepath)  # Get the directory of the Blender file
    file_path = os.path.join(blend_file_directory, filename)
    
    try:
        rated_shapes = []
        
        with open(file_path, "rb") as file:
            data = json.load(file)
            for item in data:
                shape = Shape.from_dict(item)
                rated_shapes.append(shape)
        
        print(f"FILE LOADED: '{file_path}' with {len(rated_shapes)} ratings")
        file.close()
        return rated_shapes
                
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found")
        return None
    
    
def clear_scene():
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
        
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
        

""" ------------------------------------------------------- """
""" ---------------------- OPERATORS ---------------------- """
""" ------------------------------------------------------- """

# Operator to handle the like button click
class OBJECT_OT_LikeOperator(bpy.types.Operator):
    bl_idname = "object.like_operator"
    bl_label = "Like"

    index: bpy.props.IntProperty()

    def execute(self, context):
        obj_name = f"Shape_{self.index}"
        obj = bpy.context.scene.objects.get(obj_name)

        if obj:
            obj["liked"] = 1
            self.report({'INFO'}, f"{obj_name} liked!")

        return {'FINISHED'}
 
    
# Operator to handle the dislike button click
class OBJECT_OT_DislikeOperator(bpy.types.Operator):
    bl_idname = "object.dislike_operator"
    bl_label = "Dislike"

    index: bpy.props.IntProperty()

    def execute(self, context):
        obj_name = f"Shape_{self.index}"
        obj = bpy.context.scene.objects.get(obj_name)

        if obj:
            obj["liked"] = -1
            self.report({'INFO'}, f"{obj_name} disliked!")

        return {'FINISHED'}


# Operator to generate a new set of 16 random primitive models with random colors
class OBJECT_OT_GenerateSetOperator(bpy.types.Operator):
    bl_idname = "object.generate_set_operator"
    bl_label = "Generate New Set"

    def execute(self, context):
        generate_new_set()
        return {'FINISHED'}
    
# Operator to like all shapes
class OBJECT_OT_LikeAllOperator(bpy.types.Operator):
    bl_idname = "object.like_all_operator"
    bl_label = "Like All"

    def execute(self, context):
        for i in range(grid_size*grid_size + 1):
            obj_name = f"Shape_{i}"
            obj = bpy.context.scene.objects.get(obj_name)
            if obj and obj["liked"] != -1:
                obj["liked"] = 1
                
        # Update operator properties to reflect changes in button states
        for operator in bpy.context.window_manager.operators:
            if operator.bl_idname == "object.like_operator":
                operator.depress = True
            elif operator.bl_idname == "object.dislike_operator":
                operator.depress = False
                
        self.report({'INFO'}, "All unrated shapes liked!")
        return {'FINISHED'}

# Operator to dislike all shapes
class OBJECT_OT_DislikeAllOperator(bpy.types.Operator):
    bl_idname = "object.dislike_all_operator"
    bl_label = "Dislike All"

    def execute(self, context):
        for i in range(grid_size*grid_size + 1):
            obj_name = f"Shape_{i}"
            obj = bpy.context.scene.objects.get(obj_name)
            if obj and obj["liked"] != 1:
                obj["liked"] = -1
                
        # Update operator properties to reflect changes in button states
        for operator in bpy.context.window_manager.operators:
            if operator.bl_idname == "object.like_operator":
                operator.depress = False
            elif operator.bl_idname == "object.dislike_operator":
                operator.depress = True
                
        self.report({'INFO'}, "All unrated shapes disliked!")
        return {'FINISHED'}
    
# Operator to dislike all shapes
class OBJECT_OT_ResetAllOperator(bpy.types.Operator):
    bl_idname = "object.reset_all_operator"
    bl_label = "Reset All"

    def execute(self, context):
        for i in range(grid_size*grid_size + 1):
            obj_name = f"Shape_{i}"
            obj = bpy.context.scene.objects.get(obj_name)
            if obj:
                obj["liked"] = 0
                
        # Update operator properties to reflect changes in button states
        for operator in bpy.context.window_manager.operators:
            if operator.bl_idname == "object.like_operator":
                operator.depress = False
            elif operator.bl_idname == "object.dislike_operator":
                operator.depress = False
                
        #bpy.context.area.tag_redraw()
        self.report({'INFO'}, "All shapes reset!")
        return {'FINISHED'}
    

# Custom Operator to save ratings
class OBJECT_OT_SaveRatingsOperator(bpy.types.Operator):
    bl_idname = "object.save_ratings_operator"
    bl_label = "Save Ratings"
    
    filename: bpy.props.StringProperty(
        name="Ratings filename",
        description="Enter the JSON filename",
        maxlen=1024,
        default=default_filename,
    ) # type: ignore

    def execute(self, context):
        save_shapes(self.filename)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# Custom Operator to load ratings
class OBJECT_OT_LoadRatingsOperator(bpy.types.Operator):
    global default_filename
    
    bl_idname = "object.load_ratings_operator"
    bl_label = "Load Ratings"
    
    filename: bpy.props.StringProperty(
        name="Ratings filename",
        description="Enter the JSON filename",
        maxlen=1024,
        default=default_filename,
    ) # type: ignore

    def execute(self, context):
        load_shapes(self.filename)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    

class OBJECT_OT_RunGenerationsOperator(bpy.types.Operator):
    bl_idname = "object.run_generations_operator"
    bl_label = "Run Until Convergence"
    
    convergence_treshold: bpy.props.IntProperty(
        name="Fitness treshold:",
        default=50,
        min=10,
        max=95,
        description="Enter convergence requirement in the form of a fitness treshold for the average fitness of the top 16 shapes"
        ) # type: ignore
        
    max_generations: bpy.props.IntProperty(
        name="Generations:",
        default=100,
        min=10,
        max=1000,
        description="Enter the maximum of generations that are generated to try to achieve convergence"
        ) # type: ignore

    def execute(self, context):
        run_generations(self.convergence_treshold, self.max_generations)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


# Panel to display like buttons and generate new set button
class OBJECT_PT_GAgeneratorPanel(bpy.types.Panel):
    bl_label = "GA model generator"
    bl_idname = "OBJECT_PT_GAgenerator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tools'

    def draw(self, context):
        layout = self.layout

        for i in range(grid_size*grid_size + 1):
            obj_name = f"Shape_{i}"
            obj = bpy.context.scene.objects.get(obj_name)

            if obj:
                row = layout.row()
                row.label(text=f"Shape {i}:")
                row.operator("object.like_operator", text="Like", depress = True if obj['liked'] == 1 else False).index = i
                row.operator("object.dislike_operator", text="Dislike", depress = True if obj['liked'] == -1 else False).index = i

        layout.separator()

        # Button to like all shapes that have not been rated
        layout.operator("object.like_all_operator", text="Like All Unrated")

        # Button to dislike all shapes that have not been rated
        layout.operator("object.dislike_all_operator", text="Dislike All Unrated")
        
        # Button to reset all shapes that have not been rated
        layout.operator("object.reset_all_operator", text="Clear All")
        
        layout.separator()
        
        # Button to generate a new set
        layout.operator("object.generate_set_operator", text="Generate New Set")
        
        # Button to run some number of generations
        layout.operator("object.run_generations_operator", text="Run Until Convergence")
        
        layout.separator()
        
        # Button to save ratings
        layout.operator("object.save_ratings_operator", text="Save Ratings")
        
        # Button to load ratings
        layout.operator("object.load_ratings_operator", text="Load Ratings")
        

# Register the operators and panel
def register():
    bpy.utils.register_class(OBJECT_OT_LikeOperator)
    bpy.utils.register_class(OBJECT_OT_DislikeOperator)
    bpy.utils.register_class(OBJECT_OT_GenerateSetOperator)
    bpy.utils.register_class(OBJECT_OT_LikeAllOperator)
    bpy.utils.register_class(OBJECT_OT_DislikeAllOperator)
    bpy.utils.register_class(OBJECT_OT_ResetAllOperator)
    bpy.utils.register_class(OBJECT_OT_SaveRatingsOperator)
    bpy.utils.register_class(OBJECT_OT_LoadRatingsOperator)
    bpy.utils.register_class(OBJECT_OT_RunGenerationsOperator)
    bpy.utils.register_class(OBJECT_PT_GAgeneratorPanel)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_LikeOperator)
    bpy.utils.unregister_class(OBJECT_OT_DislikeOperator)
    bpy.utils.unregister_class(OBJECT_OT_GenerateSetOperator)
    bpy.utils.unregister_class(OBJECT_OT_LikeAllOperator)
    bpy.utils.unregister_class(OBJECT_OT_DislikeAllOperator)
    bpy.utils.unregister_class(OBJECT_OT_ResetAllOperator)
    bpy.utils.unregister_class(OBJECT_OT_SaveRatingsOperator)
    bpy.utils.unregister_class(OBJECT_OT_LoadRatingsOperator)
    bpy.utils.unregister_class(OBJECT_OT_RunGenerationsOperator)
    bpy.utils.unregister_class(OBJECT_PT_GAgeneratorPanel)



if __name__ == "__main__":
    register()
    clear_scene()
    init_scene()