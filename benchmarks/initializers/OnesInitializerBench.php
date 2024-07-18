<?php
    class OnesInitializerBench
    {
        /**
        * @var string
        */
        private $shape = [];

        public function setUp(array $params): void
        {
            $this->shape = $params['shape'];
        }

        /**
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideShapes"
        * })
        */
        public function benchShapeFunctions($params): void
        {
            \NDArray::ones($this->shape);
        }

        public function provideShapes() {
            yield ['shape' => [100, 1, 1]];
            yield ['shape' => [500, 1, 1]];
            yield ['shape' => [1000, 1, 1]];
            yield ['shape' => [10, 100, 1]];
            yield ['shape' => [500, 1000, 1]];
            yield ['shape' => [1000, 10000, 1]];
        }
    }
?>
