<?php
    class FullInitializerBench
    {
        /**
        * @var shape
        */
        private $shape = [];

        public function setUp(array $params): void
        {
            $this->shape = $params['shape'];
        }

        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideShapes"
        * })
        */
        public function benchFull($params): void
        {  
            \NDArray::full($this->shape, 4);
        }

        public function provideShapes() {
            yield ['shape' => [100, 1]];
            yield ['shape' => [500, 1]];
            yield ['shape' => [1000, 1]];
            yield ['shape' => [10, 100]];
            yield ['shape' => [500, 1000]];
            yield ['shape' => [1000, 10000]];
        }
    }
?>
