import configparser
import ast

from pygame_renderer import Pygame_Renderer
from sns_renderer import SNS_Renderer
from sns_renderer2 import SNS_Renderer2
from bio_environment import BioEnvironment
from bio_world import BioGymWorld


class ConfigParser:

    def __init__(self, config_filename) -> None:
        self.config = configparser.ConfigParser(inline_comment_prefixes=";")
        self.config.read(config_filename)

    def create_renderer(self, num_species, grid_size, prot_unit_size):
        renderer_type = self.config["Renderer"]["renderer_type"]
        render_mode = self.config["Renderer"]["render_mode"]
        sim_height = int(self.config["Renderer"]["sim_height"])
        pix_padding = int(self.config["Renderer"]["render_pix_padding"])
        display_population = self.config["Renderer"].getboolean(
            "display_population")

        renderer = self.parse_renderer_type(renderer_type)

        return renderer(render_mode, sim_height, pix_padding, num_species,
                        grid_size, prot_unit_size, display_population)

    def create_bio_environment(self, num_species, grid_size,
                               prot_unit_size) -> BioEnvironment:
        diagonal_neighbours = self.config["BioEnvironment"].getboolean(
            "diagonal_neighbours")
        migration_rate = ast.literal_eval(
            self.config["BioEnvironment"]["migration_rate"])
        species_ranges = ast.literal_eval(
            self.config["BioEnvironment"]["species_ranges"])
        extinction_threshold = ast.literal_eval(
            self.config["BioEnvironment"]["extinction_threshold"])

        r = self.config["BioEnvironment"].getfloat("r")
        k = self.config["BioEnvironment"].getfloat("k")
        a = self.config["BioEnvironment"].getfloat("a")
        b = self.config["BioEnvironment"].getfloat("b")
        e = self.config["BioEnvironment"].getfloat("e")
        d = self.config["BioEnvironment"].getfloat("d")
        a_2 = self.config["BioEnvironment"].getfloat("a_2")
        b_2 = self.config["BioEnvironment"].getfloat("b_2")
        e_2 = self.config["BioEnvironment"].getfloat("e_2")
        d_2 = self.config["BioEnvironment"].getfloat("d_2")
        s = self.config["BioEnvironment"].getfloat("s")
        gamma = self.config["BioEnvironment"].getfloat("gamma")

        return BioEnvironment(num_species, grid_size, prot_unit_size,
                              diagonal_neighbours, migration_rate,
                              species_ranges, extinction_threshold, r, k, a, b,
                              e, d, a_2, b_2, e_2, d_2, s, gamma)

    def create_bio_gym_world(self) -> BioGymWorld:
        num_species = int(self.config["BioGymWorld"]["num_species"])
        grid_size = int(self.config["BioGymWorld"]["grid_size"])
        prot_unit_size = int(self.config["BioGymWorld"]["prot_unit_size"])

        bio_env = self.create_bio_environment(num_species, grid_size,
                                              prot_unit_size)
        renderer = self.create_renderer(num_species, grid_size, prot_unit_size)

        return BioGymWorld(bio_env, renderer)

    def parse_renderer_type(self, renderer_type):
        if renderer_type == "pygame":
            return Pygame_Renderer
        elif renderer_type == "sns":
            return SNS_Renderer
        elif renderer_type == "sns2":
            return SNS_Renderer2
        else:
            raise NotImplementedError("Renderer type " + str(renderer_type) +
                                      " not implemented.")
