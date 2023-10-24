import unittest
import torch

from train import load_data
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.discriminator_values = torch.rand((64, 3, 64, 64))
        self.discriminator = Discriminator(3, 16)
        self.generator_values = torch.rand((64, 100, 1, 1))
        self.generator = Generator(100, 3, 16)
        self.real_values = torch.rand((64, 3, 64, 64))
        self.fake_values = self.generator(self.generator_values)

    # test discriminator

    def test_disc_is_correct(self):
        self.assertEqual(self.discriminator(self.discriminator_values).shape, torch.Size((64, 1, 1, 1)))

    def test_disc_output_is_not_string(self):
        self.assertNotEqual(self.discriminator(self.discriminator_values).shape, "")

    def test_disc_output_is_correct_shape(self):
        self.assertNotEqual(self.discriminator(self.discriminator_values).shape, torch.Size((64, 3, 64, 64)))

    # test generator

    def test_gen_is_correct(self):
        self.assertEqual(self.generator(self.generator_values).shape, torch.Size((64, 3, 64, 64)))

    def test_gen_output_is_not_string(self):
        self.assertNotEqual(self.generator(self.generator_values).shape, "")

    def test_gen_output_is_correct_shape(self):
        self.assertNotEqual(self.generator(self.generator_values).shape, torch.Size((64, 1, 1, 1)))

    # test initialize_weights

    def test_initialize_weights(self):
        self.assertEqual(initialize_weights(self.discriminator), None)
        self.assertEqual(initialize_weights(self.generator), None)

    # test gradient_penalty

    def test_gradient_penalty(self):
        self.assertTrue(isinstance(
            gradient_penalty(self.discriminator.to("cuda"), self.real_values.cuda(), self.fake_values.cuda(), "cuda").data.item(),
            float), "Result is not a float")

        self.assertFalse(isinstance(
            gradient_penalty(self.discriminator.to("cuda"), self.real_values.cuda(), self.fake_values.cuda(), "cuda").data.item(),
            str), "Result is a float")


if __name__ == '__main__':
    unittest.main()
