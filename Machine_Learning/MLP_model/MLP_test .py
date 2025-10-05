import unittest
import numpy as np
import math

from keras.src.optimizers import SGD

from MLP import FC, MLP, CrossEntropyLoss, SGDOptimizer


class TestFCMethods(unittest.TestCase):
    def test_fc_init(self):
        fc = FC(n_in=10,n_out=5,activation = "sigmoid")
        self.assertEqual(fc.n_in,10)
        self.assertEqual(fc.n_out,5)
        self.assertEqual(fc.W.shape,(10,5))
        self.assertEqual(fc.b.shape,(1,5))
        self.assertEqual(fc.dW.shape,(10,5))
        self.assertEqual(fc.activation , "sigmoid")
    def test_fc_forward(self):
        fc = FC(n_in=10,n_out=5,activation = "sigmoid")
        x = np.zeros(shape=(3,10),dtype=np.float32)
        y= fc.forward(x)
        # sigmoid(0) = 0.5
        self.assertTrue(np.allclose(y, 0.5 * np.ones((3, 5)), atol=1e-6))
    def test_fc_forward_identity(self):
        fc = FC(n_in=10,n_out=5,activation = None)
        x = np.zeros(shape=(3, 10), dtype=np.float32)
        y = fc.forward(x)
        # output phải toàn 0
        self.assertTrue(np.allclose(y, np.zeros((3, 5)), atol=1e-6))
    def test_fc_backward_identity(self):
        fc = FC(n_in=10,n_out=5,activation = None)
        x = np.zeros(shape=(3, 10), dtype=np.float32)
        y = fc.forward(x)
        dx = fc.backward(np.ones_like(y))
        self.assertEqual(dx.shape,x.shape)
        self.assertEqual(fc.dW.shape,fc.W.shape)
class TestMLPMethods(unittest.TestCase):
    def test_mlp_init(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        layer0 = model.layers[0]
        layer1 = model.layers[1]
        self.assertEqual(model.n_input,10)
        self.assertEqual(model.hiddens,[5,2])
        self.assertEqual(layer0.n_in,10)
        self.assertEqual(layer0.n_out,5)
        self.assertEqual(layer1.n_in,5)
        self.assertEqual(layer1.n_out,2)
    def test_mlp_forward(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        x = np.zeros(shape=(3,10), dtype=np.float32)
        y= model.forward(x)
        self.assertEqual(y.shape,(3,2))
    def test_mlp_backward(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        x = np.zeros(shape=(3, 10), dtype=np.float32)
        y= model.forward(x)
        dx = model.backward(np.ones_like(y))
        self.assertEqual(dx.shape,x.shape)
class TestCEMethods(unittest.TestCase):
    def test_ce_forward(self):
        ypred = np.zeros((10,5))
        ytrue = np.array([0,1,2,3,4,0,1,2,3,4],dtype = int)
        ce = CrossEntropyLoss()
        loss = ce.forward(ypred, ytrue)
        self.assertAlmostEqual(loss,-10*math.log(1/5))
    def test_ce_backward(self):
        ypred = np.zeros((10,5))
        ytrue = np.array([0,1,2,3,4,0,1,2,3,4],dtype = int)
        ce = CrossEntropyLoss()
        loss = ce.forward(ypred, ytrue)
        d_ypred = ce.backward()
        desired = np.ones((10,5))*0.2
        desired[range(10),ytrue] -= 1
        error = np.sum(np.abs(desired - d_ypred))
        self.assertAlmostEqual(error,0)
class TestSGDMethods(unittest.TestCase):
    def test_sgd_init(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        sgd = SGDOptimizer(model,learning_rate = 0.2)
        param = sgd.parameters()
        grad = sgd.grads()
        for p,g in zip(param,grad):
            self.assertEqual(p.shape,g.shape)
        self.assertEqual(sgd.learning_rate, 0.2)
    def test_sgd_zero_grad(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        sgd = SGDOptimizer(model,learning_rate = 0.2)
        sgd.zero_grad()
        for g in sgd.grads():
            self.assertAlmostEqual(np.sum(np.abs(g)),0)
    def test_sgd_step(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        sgd = SGDOptimizer(model,learning_rate = 0.2)
        loss_func = CrossEntropyLoss()
        x = np.zeros(shape=(3,10), dtype=np.float32)
        ytrue = np.array([0,1,0],dtype = int)
        ypred = model.forward(x)
        loss = loss_func.forward(ypred, ytrue)
        sgd.zero_grad()
        dout = loss_func.backward()
        dx = model.backward(dout)
        sgd.step()
    def test_sgd_n_step(self):
        model = MLP(n_input = 10, hiddens = [5,2])
        sgd = SGDOptimizer(model,learning_rate = 0.2)
        loss_func = CrossEntropyLoss()
        n_steps = 10
        x= np.zeros(shape=(3,10), dtype=np.float32)
        ytrue = np.array([0,1,0],dtype = int)
        print()
        for step in range(n_steps):
            ypred = model.forward(x)
            loss = loss_func.forward(ypred, ytrue)
            print(f"step: {step}, loss: {loss}")
            if step > 0:
                self.assertLess(loss,old_loss)
            old_loss = loss
            sgd.zero_grad()
            dout = loss_func.backward()
            dx = model.backward(dout)
            sgd.step()
if __name__ == '__main__':
    unittest.main()
