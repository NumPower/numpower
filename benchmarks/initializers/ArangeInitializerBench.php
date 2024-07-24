<?php
    class ArangeInitializerBench
    {
        /**
        * @var size
        */
        private $size = 1;

        public function setUp(array $params): void
        {
            $this->size = $params['size'];
        }
        
        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideSizes",
        * })
        */
        public function benchARange($params)
        {
            $ndarray = \NDArray::arange($this->size, 0, 1);
        }
        public function provideSizes() {
            yield ['size' => 100];
            yield ['size' => 500];
            yield ['size' => 1000];
        }
    }
?>